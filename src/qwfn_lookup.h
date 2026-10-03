#pragma once
// Prompt-lookup drafts: when the reply copies its context (code edits, quoted files, repeated structure), the tokens
// that followed an earlier occurrence of the current suffix are usually the next ones, and a verify step can accept
// several at once. No model, microseconds per step.
//
// lookup_drafter is a port of Strata's suffix drafter (github.com/Niko1221/Strata, src/spec/suffix_drafter.cpp,
// MIT): every trigram of the history maps to its WAYS most recent end positions in a fixed-size open-addressing
// table; a proposal extends each candidate match backwards up to max_match and takes the longest (the most recent
// on ties). lookup_policy follows Strata's draft policy (expected committed tokens per millisecond, lookup
// acceptance learned per match-length bucket, measured step cost per window size), with one change for this engine:
// a window size other than the head's costs a rebuild of the layer graphs (the device memory does not hold a graph
// set per size), so the policy switches into a lookup window only on a strong match and stays there while the copy
// lasts (hysteresis), instead of choosing per step. Measured (forced lookup windows on file edits and
// quotes): the head already drafts copied text at ~99%, a 4-position step costs ~1.65x a head step, lookup steps
// commit ~3.6 tokens: +2-13% on copying replies. The first steps of a new window size are warm-up (one took 250 ms)
// and are not costs: each size's first WARMUP steps after a change are skipped.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qwfn {

class lookup_drafter {
public:
    static constexpr int WAYS = 4;

    explicit lookup_drafter(int min_match = 3, int max_match = 32);

    // Empty history, table sized for about `capacity_tokens` (load factor <= 0.5 there; past it, new trigrams are
    // not indexed and older matches still work).
    void reset(size_t capacity_tokens);
    void append(const int32_t * tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }
    // Up to max_k tokens that followed the longest earlier match of the history's suffix; 0 = no match of at least
    // min_match tokens.
    int propose(int max_k, int32_t * out);
    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }
    const std::vector<int32_t> & history() const { return hist_; }

private:
    struct slot {
        uint64_t key = 0;            // trigram hash, never 0 (0 = empty)
        uint32_t pos[WAYS] = {};     // end positions, most recent first
        uint8_t  n = 0;
    };
    slot * find_slot(uint64_t key, bool insert);
    uint64_t key_at(size_t end) const;

    int min_match_, max_match_;
    std::vector<int32_t> hist_;
    std::vector<slot> table_;
    size_t mask_ = 0;
    int last_match_ = 0;
};

// Per step: the head's window (T = 2: one draft) or the lookup window (T = 1 + k_max), with hysteresis.
class lookup_policy {
public:
    static constexpr int MAX_T = 4;
    static constexpr int BUCKETS = 4;

    struct params {
        int   k_max = 3;           // lookup drafts per step (the window is 1 + k_max)
        int   enter_match = 12;    // a match at least this long to switch into the lookup window
        int   patience = 2;        // consecutive steps where the lookup window does not pay before switching back
        float margin = 0.15f;      // the lookup window's tokens/ms must beat the head's by this much to switch in
        bool  force = false;       // diagnostic: a lookup window whenever there is a proposal (measures its cost)
        int   head_t = 2;          // the head's window (1 + its drafts). When it equals the lookup window (1 + k_max),
                                   // nothing is rebuilt by a switch: each step simply takes the source expected to
                                   // commit more tokens (no hysteresis, no entry threshold beyond min_match)
    };
    explicit lookup_policy(params p);

    // This step's window: true = lookup (T = 1 + k_max), false = the head's. `k`: the proposal's length (0 = none),
    // `match`: its match length.
    bool choose(int k, int match);
    bool in_lookup() const { return lookup_; }
    // A new request starts on the head's window (its history is new); the learned costs and acceptance stay.
    void new_sequence() { lookup_ = false; bad_steps_ = 0; }
    // After the step: the window it used, the real lookup drafts it carried (not the padding) and how many of them
    // were accepted (lookup), or the drafts accepted (head); its wall time; and whether its layer graphs were
    // rebuilt (then the time is not a window cost).
    void observe(bool lookup, int t, int drafted, int accepted, int match, double step_ms, bool rebuilt);

    double rate_lookup(int k, int match) const;   // expected tokens per ms of a lookup step
    double rate_head() const;                     // of a head step
    double expected_lookup(int k, int match) const;   // tokens a lookup step of k real drafts commits
    bool   same_size() const { return p_.head_t == 1 + p_.k_max; }
    double cost_ms(int t) const;                  // measured (or scaled) step time of a window of t positions
    double lookup_q(int match) const;             // a lookup draft's acceptance for this match length
    // One line: measured step cost per window size (samples), head tokens per step, q per bucket, switches.
    std::string describe() const;
    long entries() const { return entries_; }

private:
    static int bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }

    params p_;
    bool lookup_ = false;
    int  bad_steps_ = 0;
    std::array<double, MAX_T + 1> cost_{}, cost_n_{};
    double head_tok_ = 1.8, head_n_ = 0;            // tokens committed per head step (EMA)
    std::array<double, BUCKETS> ok_{}, bad_{};      // lookup drafts accepted / rejected, decayed
    long entries_ = 0, exits_ = 0, rebuilt_ = 0, skipped_ = 0;
    int  last_t_ = 0, warm_left_ = 0;
    std::array<int, MAX_T + 1> outliers_{};       // consecutive rejected samples per size: 4 in a row = a new regime
};

}  // namespace qwfn
