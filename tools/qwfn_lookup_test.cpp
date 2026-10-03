// qwfn-lookup-test: unit tests of the prompt-lookup drafter and its window policy, and an offline simulation.
//
//   qwfn-lookup-test                                   unit tests
//   qwfn-lookup-test --simulate [--k K] FILE.ids ...   (token ids, whitespace- or comma-separated)
//
// The simulation replays a token sequence as if the model produced it, with a stand-in for the draft head (its draft
// is right with probability HEAD_P) and a step-cost model (ms per window size, plus a graph rebuild whenever the
// window size changes), and compares tokens per second of the head alone against the head plus lookup windows. Two
// regimes per file, as Strata's suffix_drafter_test: `continue` (history = the first half, generate the second) and
// `copy` (history = the whole sequence, generate its middle third again: output quoting its input).
#include "qwfn_lookup.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using qwfn::lookup_drafter;
using qwfn::lookup_policy;

namespace {
int g_fail = 0;
void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char * path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char & c : text) if (c == ',' || c == '[' || c == ']') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

void drafter_tests() {
    {   // a repeated passage is proposed verbatim
        lookup_drafter d;
        const std::vector<int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 11, 12};
        d.append(doc.data(), doc.size());
        int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    {   // no earlier occurrence: nothing proposed
        lookup_drafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) == 0 && d.last_match() == 0, "no match proposes nothing");
    }
    {   // the longer match wins even though it is older
        lookup_drafter d;
        const std::vector<int32_t> doc = {7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    {   // a match shorter than min_match is ignored
        lookup_drafter d(4);
        const std::vector<int32_t> doc = {1, 2, 3, 9, 5, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    {   // periodic text: the continuation may run into the current suffix
        lookup_drafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    {   // past the nominal capacity: no crash, still proposes
        lookup_drafter d;
        d.reset(1024);
        std::vector<int32_t> doc;
        for (int i = 0; i < 50000; ++i) doc.push_back(i % 997);
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflowing the nominal capacity");
    }
    {   // reset forgets everything
        lookup_drafter d;
        const std::vector<int32_t> doc = {5, 6, 7, 8, 5, 6, 7};
        d.append(doc.data(), doc.size());
        d.reset(64);
        int32_t out[2];
        check(d.size() == 0 && d.propose(2, out) == 0, "reset clears the history and the index");
    }
    {   // short histories and max_k <= 0
        lookup_drafter d;
        int32_t out[2];
        check(d.propose(2, out) == 0, "empty history");
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        check(d.propose(0, out) == 0, "max_k 0");
    }
}

void policy_tests() {
    lookup_policy::params p; p.k_max = 3; p.enter_match = 6; p.patience = 2; p.margin = 0.10f;
    {   // a weak match does not switch in; a strong full proposal does
        lookup_policy pol(p);
        for (int i = 0; i < 20; ++i) { pol.observe(false, 2, 1, 1, 0, 36.0, false); }
        check(!pol.choose(3, 3), "match below enter_match stays on the head");
        check(!pol.choose(2, 20), "a partial proposal does not switch in");
        check(pol.choose(3, 20), "a full proposal from a long match switches in");
        check(pol.in_lookup(), "the switch is remembered");
        check(pol.choose(3, 20), "a strong full proposal keeps the window without the entry margin");
    }
    {   // patience: two bad steps switch back out
        lookup_policy pol(p);
        for (int i = 0; i < 20; ++i) pol.observe(false, 2, 1, 1, 0, 36.0, false);
        check(pol.choose(3, 30), "in");
        check(pol.choose(0, 0), "one step without a proposal keeps the window (patience)");
        check(!pol.choose(0, 0), "the second one switches back");
    }
    {   // learned acceptance: a bucket whose drafts keep failing stops paying
        lookup_policy pol(p);
        for (int i = 0; i < 20; ++i) pol.observe(false, 2, 1, 1, 0, 36.0, false);
        for (int i = 0; i < 60; ++i) pol.observe(true, 4, 3, 0, 30, 46.0, false);
        check(pol.lookup_q(30) < 0.2, "q learned down");
        check(!pol.choose(3, 30), "a failing bucket does not switch in");
    }
    {   // a new sequence starts on the head's window, keeping what was learned
        lookup_policy pol(p);
        for (int i = 0; i < 20; ++i) pol.observe(false, 2, 1, 1, 0, 36.0, false);
        check(pol.choose(3, 30), "in");
        const double q = pol.lookup_q(30);
        pol.new_sequence();
        check(!pol.in_lookup() && pol.lookup_q(30) == q, "new_sequence: head window, learned state kept");
    }
    {   // same window size as the head: per step, the source expected to commit more; no hysteresis
        lookup_policy::params q = p; q.k_max = 2; q.head_t = 3;
        lookup_policy pol(q);
        for (int i = 0; i < 20; ++i) pol.observe(false, 3, 2, 1, 0, 45.0, false);   // the head commits ~2 per step
        check(pol.same_size(), "same size");
        check(pol.choose(2, 30), "a strong 2-token proposal beats the head's ~2 tokens");
        check(!pol.choose(0, 0), "no proposal: the head, immediately (no patience)");
        check(pol.choose(2, 30), "and back the next step");
    }
    {   // rebuilt and warm-up steps are not window costs
        lookup_policy pol(p);
        for (int i = 0; i < 12; ++i) pol.observe(false, 2, 1, 1, 0, 36.0, false);   // past the head's own warm-up
        pol.observe(true, 4, 3, 3, 30, 500.0, true);
        check(pol.cost_ms(4) < 100.0, "a rebuild step's time is ignored");
        for (int i = 0; i < 5; ++i) pol.observe(true, 4, 3, 3, 30, 150.0, false);   // warm-up steps (measured: ~150 ms)
        check(pol.cost_ms(4) < 100.0, "warm-up steps' times are ignored");
        const double prior4 = pol.cost_ms(4);
        pol.observe(true, 4, 3, 3, 30, 60.0, false);
        const double c1 = pol.cost_ms(4);
        check(c1 != 60.0 && c1 != prior4 && (c1 - 60.0) * (c1 - prior4) < 0, "one steady sample moves the estimate toward it, not onto it");
        for (int i = 0; i < 10; ++i) pol.observe(true, 4, 3, 3, 30, 60.0, false);
        check(pol.cost_ms(4) == 60.0, "enough steady samples: the measured cost");
        pol.observe(true, 4, 3, 3, 30, 900.0, false);
        check(pol.cost_ms(4) == 60.0, "a step past 3x the established cost is ignored");
        pol.observe(true, 4, 3, 3, 30, 900.0, false);
        pol.observe(true, 4, 3, 3, 30, 900.0, false);
        check(pol.cost_ms(4) == 60.0, "three in a row: still noise");
        pol.observe(true, 4, 3, 3, 30, 900.0, false);
        check(pol.cost_ms(4) > 60.0, "the fourth in a row: a new regime, the estimate restarts there");
    }
}

struct sim_result { double ms = 0; long tokens = 0, steps = 0, lookup_steps = 0, switches = 0, lookup_acc = 0, lookup_drafted = 0; };

// cost[t]: ms of a step with t positions; rebuild_ms: the extra time of a step whose window size changed.
sim_result simulate(const std::vector<int32_t> & history, const std::vector<int32_t> & target, bool use_lookup,
                    int k_max, double head_p, const double * cost, double rebuild_ms, unsigned seed) {
    lookup_drafter d;
    d.reset(history.size() + target.size() + 16);
    d.append(history.data(), history.size());
    lookup_policy::params pp; pp.k_max = k_max;
    if (const char * v = std::getenv("QWFN_LOOKUP_ENTER"))    pp.enter_match = std::atoi(v);
    if (const char * v = std::getenv("QWFN_LOOKUP_PATIENCE")) pp.patience = std::atoi(v);
    if (const char * v = std::getenv("QWFN_LOOKUP_MARGIN"))   pp.margin = (float) std::atof(v);
    lookup_policy pol(pp);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    sim_result r;
    int last_t = 2;
    std::vector<int32_t> prop(k_max);
    size_t i = 0;
    while (i < target.size()) {
        int k = 0, match = 0;
        bool lk = false;
        if (use_lookup) { k = d.propose(k_max, prop.data()); match = d.last_match(); lk = pol.choose(k, match); }
        const int t = lk ? 1 + k_max : 2;
        int acc = 0, drafted = 0;
        if (lk) {
            drafted = k;
            while (acc < k && i + acc < target.size() && prop[acc] == target[i + acc]) ++acc;
        } else {
            drafted = 1;
            acc = (i + 1 < target.size() && U(rng) < head_p) ? 1 : 0;
        }
        const size_t commit = std::min(target.size() - i, (size_t) acc + 1);
        const bool rebuilt = t != last_t;
        const double ms = cost[t] + (rebuilt ? rebuild_ms : 0.0);
        pol.observe(lk, t, drafted, acc, match, ms, rebuilt);
        d.append(&target[i], commit);
        i += commit;
        r.tokens += (long) commit; r.steps++; r.ms += ms;
        if (lk) { r.lookup_steps++; r.lookup_drafted += drafted; r.lookup_acc += acc; }
        if (rebuilt) r.switches++;
        last_t = t;
    }
    return r;
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int K = 3, first = 2;
        if (argc > 3 && std::strcmp(argv[2], "--k") == 0) { K = std::atoi(argv[3]); first = 4; }
        // This engine: a T=2 step ~37 ms, T=3 ~1.30x and T=4 ~1.65x of it on copied text (forced lookup
        // windows); a window change rebuilds 48 layer graphs, ~0.55 ms each. The head's draft
        // is right with probability QWFN_SIM_HEAD_P (default 0.80; ~0.99 measured on copied text).
        const double cost[5] = {0, 26.0, 37.0, 48.1, 61.0};
        const double rebuild = 26.0, head_p = std::getenv("QWFN_SIM_HEAD_P") ? std::atof(std::getenv("QWFN_SIM_HEAD_P")) : 0.80;
        std::printf("%-26s %7s | %-40s | %-40s\n", "file", "tokens", "continue: head  +lookup  (lk steps, sw)",
                    "copy: head  +lookup  (lk steps, sw)");
        for (int f = first; f < argc; ++f) {
            const std::vector<int32_t> ids = read_ids(argv[f]);
            if (ids.size() < 64) continue;
            const size_t half = ids.size() / 2, a = ids.size() / 3, b = 2 * ids.size() / 3;
            const std::vector<int32_t> h1(ids.begin(), ids.begin() + half), t1(ids.begin() + half, ids.end());
            const std::vector<int32_t> t2(ids.begin() + a, ids.begin() + b);
            const sim_result c0 = simulate(h1, t1, false, K, head_p, cost, rebuild, 1), c1 = simulate(h1, t1, true, K, head_p, cost, rebuild, 1);
            const sim_result p0 = simulate(ids, t2, false, K, head_p, cost, rebuild, 1), p1 = simulate(ids, t2, true, K, head_p, cost, rebuild, 1);
            const char * name = std::strrchr(argv[f], '/') ? std::strrchr(argv[f], '/') + 1 : argv[f];
            auto tps = [](const sim_result & r) { return 1000.0 * r.tokens / r.ms; };
            std::printf("%-26s %7zu | %6.1f %6.1f %+5.1f%% (%4.1f%%, %ld) | %6.1f %6.1f %+5.1f%% (%4.1f%%, %ld)\n", name, ids.size(),
                        tps(c0), tps(c1), 100.0 * (tps(c1) / tps(c0) - 1), 100.0 * c1.lookup_steps / c1.steps, c1.switches,
                        tps(p0), tps(p1), 100.0 * (tps(p1) / tps(p0) - 1), 100.0 * p1.lookup_steps / p1.steps, p1.switches);
        }
        return 0;
    }
    drafter_tests();
    policy_tests();
    std::printf("lookup drafter / policy tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
