#include "qwfn_lookup.h"

#include <algorithm>
#include <cstdio>

namespace qwfn {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

lookup_drafter::lookup_drafter(int min_match, int max_match)
    : min_match_(std::max(3, min_match)), max_match_(std::max(std::max(3, min_match), max_match)) {
    reset(1u << 12);
}

void lookup_drafter::reset(size_t capacity_tokens) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;
    table_.assign(cap, slot{});
    mask_ = cap - 1;
    hist_.clear();
    hist_.reserve(capacity_tokens);
    last_match_ = 0;
}

uint64_t lookup_drafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;
}

lookup_drafter::slot * lookup_drafter::find_slot(uint64_t key, bool insert) {
    // Bounded probing: a full table returns nullptr rather than scanning it all on every lookup.
    for (size_t i = key & mask_, probes = 0; probes <= mask_ && probes < 64; i = (i + 1) & mask_, ++probes) {
        slot & s = table_[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            return &s;
        }
    }
    return nullptr;
}

void lookup_drafter::append(const int32_t * tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hist_.push_back(tokens[i]);
        const size_t end = hist_.size() - 1;
        if (end < 2 || end > UINT32_MAX) continue;
        slot * s = find_slot(key_at(end), true);
        if (!s) continue;
        for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS) ++s->n;
    }
}

int lookup_drafter::propose(int max_k, int32_t * out) {
    last_match_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    const slot * s = find_slot(key_at(cur), false);
    if (!s) return 0;
    size_t best_end = 0;
    int best_len = 0;
    for (int w = 0; w < s->n; ++w) {
        const size_t p = s->pos[w];
        if (p >= cur) continue;   // the current suffix itself
        int len = 0;
        while (len < max_match_ && (size_t) len <= p && hist_[p - len] == hist_[cur - len]) ++len;
        if (len > best_len) { best_len = len; best_end = p; }   // most recent first: ties keep the newer
    }
    if (best_len < min_match_) return 0;
    last_match_ = best_len;
    int k = 0;
    // The continuation may run into the current suffix (periodic text): history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

namespace {
// A step's cost by window size relative to one position, for sizes without enough steady samples: T=2 1.45x
// (measured); T=3 1.22x and T=4 1.55x of a T=2 step in steady state (forced lookup windows).
constexpr double SHAPE[lookup_policy::MAX_T + 1] = {0.0, 1.0, 1.45, 1.77, 2.25};
constexpr double COST_ALPHA = 0.1, TOK_ALPHA = 0.05, DECAY = 0.97;
constexpr int    WARMUP = 6;          // steps of a window size not counted as its cost after a size change (the
                                      // first few T=3 steps after a switch ran ~150 ms against ~67 steady)
constexpr double TRUST_N = 8.0;       // steady samples before a size's measured cost fully replaces the scaled prior
constexpr double OUTLIER = 3.0;       // a step slower than this x the established cost is not a cost sample...
constexpr int    REGIME = 4;          // ...unless this many in a row: the context changed (a long prompt), not noise
// Before a bucket has data: the longer the match, the likelier its continuation (Strata's priors), worth 4 drafts.
constexpr double PRIOR_Q[lookup_policy::BUCKETS] = {0.75, 0.88, 0.93, 0.96};
constexpr double PRIOR_N = 4.0;
}  // namespace

lookup_policy::lookup_policy(params p) : p_(p) {
    p_.k_max = std::clamp(p_.k_max, 1, MAX_T - 1);
    p_.patience = std::max(1, p_.patience);
    p_.head_t = std::clamp(p_.head_t, 2, MAX_T);
}

double lookup_policy::lookup_q(int match) const {
    const int b = bucket(match);
    return (ok_[b] + PRIOR_N * PRIOR_Q[b]) / (ok_[b] + bad_[b] + PRIOR_N);
}

double lookup_policy::cost_ms(int t) const {
    t = std::clamp(t, 1, MAX_T);
    // The prior: this size scaled from the sizes measured (weighted by their samples), excluding itself.
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= MAX_T; ++u)
        if (u != t && cost_n_[u] > 0) {
            const double w = std::min(cost_n_[u], 20.0);
            num += w * cost_[u] * SHAPE[t] / SHAPE[u];
            den += w;
        }
    const double prior = den > 0 ? num / den : SHAPE[t];
    if (cost_n_[t] <= 0) return prior;
    if (den <= 0) return cost_[t];
    // A few samples do not overrule the prior (a misread warm-up would lock a size out for good).
    const double w = std::min(1.0, cost_n_[t] / TRUST_N);
    return w * cost_[t] + (1.0 - w) * prior;
}

double lookup_policy::rate_head() const { return head_tok_ / cost_ms(p_.head_t); }

double lookup_policy::expected_lookup(int k, int match) const {
    // E = 1 + q + q^2 + ... + q^k committed tokens (a rejection ends the step with the trunk's own token).
    k = std::clamp(k, 0, p_.k_max);
    const double q = lookup_q(match);
    double e = 1.0, qi = 1.0;
    for (int i = 0; i < k; ++i) { qi *= q; e += qi; }
    return e;
}

double lookup_policy::rate_lookup(int k, int match) const {
    // At the cost of the full lookup window (the padding positions cost the same as real ones).
    return expected_lookup(k, match) / cost_ms(1 + p_.k_max);
}

bool lookup_policy::choose(int k, int match) {
    if (same_size()) {
        // One window size for both sources: per step, the one expected to commit more tokens.
        lookup_ = k > 0 && (p_.force || expected_lookup(k, match) > head_tok_);
        return lookup_;
    }
    const bool pays = k > 0 && (p_.force || rate_lookup(k, match) > rate_head() * (lookup_ ? 1.0 : 1.0 + p_.margin));
    if (!lookup_) {
        // In only on a full proposal from a strong match: a window change rebuilds the layer graphs.
        if (pays && k >= p_.k_max && match >= p_.enter_match) { lookup_ = true; bad_steps_ = 0; entries_++; }
    } else if (pays) {
        bad_steps_ = 0;
    } else if (++bad_steps_ >= p_.patience) {
        lookup_ = false; bad_steps_ = 0; exits_++;
    }
    return lookup_;
}

void lookup_policy::observe(bool lookup, int t, int drafted, int accepted, int match, double step_ms, bool rebuilt) {
    t = std::clamp(t, 1, MAX_T);
    if (rebuilt) rebuilt_++;
    if (t != last_t_) { last_t_ = t; warm_left_ = WARMUP; }
    bool sample = step_ms > 0 && !rebuilt;
    if (warm_left_ > 0) { warm_left_--; sample = false; skipped_++; }
    if (sample && cost_n_[t] >= 1 && step_ms > OUTLIER * cost_[t]) {
        if (++outliers_[t] < REGIME) { sample = false; skipped_++; }
        else { outliers_[t] = 0; cost_[t] = step_ms; cost_n_[t] = 1; sample = false; }   // restart the estimate here
    } else if (sample) {
        outliers_[t] = 0;
    }
    if (sample) {
        cost_[t] = cost_n_[t] > 0 ? (1.0 - COST_ALPHA) * cost_[t] + COST_ALPHA * step_ms : step_ms;
        cost_n_[t] += 1.0;
    }
    if (lookup) {
        if (drafted > 0) {
            const int b = bucket(match);
            ok_[b]  = DECAY * ok_[b]  + accepted;
            bad_[b] = DECAY * bad_[b] + (accepted < drafted ? 1.0 : 0.0);
        }
    } else if (t == p_.head_t) {
        const double got = accepted + 1.0;
        head_tok_ = head_n_ > 0 ? (1.0 - TOK_ALPHA) * head_tok_ + TOK_ALPHA * got : got;
        head_n_ += 1.0;
    }
}

std::string lookup_policy::describe() const {
    char b[512];
    int n = snprintf(b, sizeof b, "cost ms by T:");
    for (int t = 1; t <= MAX_T; ++t)
        n += snprintf(b + n, sizeof b - n, " %d:%.1f(%.0f)", t, cost_ms(t), cost_n_[t]);
    n += snprintf(b + n, sizeof b - n, " | head (T=%d) %.2f tok/step | q by match bucket:", p_.head_t, head_tok_);
    for (int i = 0; i < BUCKETS; ++i) n += snprintf(b + n, sizeof b - n, " %.2f", lookup_q(i == 0 ? 3 : i == 1 ? 6 : i == 2 ? 12 : 24));
    snprintf(b + n, sizeof b - n, " | entries %ld, exits %ld, rebuilt steps %ld, skipped samples %ld%s", entries_, exits_, rebuilt_, skipped_,
             p_.force ? " (forced)" : "");
    return b;
}

}  // namespace qwfn
