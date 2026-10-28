"""Our fixed checker of problem 112 (flint/checkers/112/chk.cc, audit 2026-10-28): a fake output that the official
checker scores 100 (an infinite claimed distance, every point at the origin) scores 0, an output with a nan
coordinate scores 0, and a real program scores the same in every case under the official and the fixed checker."""
from pathlib import Path

from helpers import check, done, env

env()
from flint import judge as J  # noqa: E402

P = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/112"
FAKE_INF = r'''#include <cstdio>
int main() { int n; if (scanf("%d", &n) != 1) return 0; printf("1e400\n"); for (int i = 0; i < n; i++) printf("0 0 0\n"); }
'''
FAKE_NAN = r'''#include <cstdio>
int main() { int n; if (scanf("%d", &n) != 1) return 0; printf("1e18\n"); for (int i = 0; i < n; i++) printf("nan nan nan\n"); }
'''
REAL = r'''#include <bits/stdc++.h>
int main() {   // a Fibonacci sphere, with its exact minimum distance
    int n; if (scanf("%d", &n) != 1) return 0;
    std::vector<double> x(n), y(n), z(n);
    const double g = M_PI * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < n; i++) {
        double t = n == 1 ? 0.0 : 1.0 - 2.0 * i / (n - 1), r = std::sqrt(std::max(0.0, 1.0 - t * t));
        x[i] = r * std::cos(g * i); y[i] = r * std::sin(g * i); z[i] = t;
    }
    double best = 1e18;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
        best = std::min(best, std::sqrt((x[i]-x[j])*(x[i]-x[j]) + (y[i]-y[j])*(y[i]-y[j]) + (z[i]-z[j])*(z[i]-z[j])));
    printf("%.12f\n", best);
    for (int i = 0; i < n; i++) printf("%.12f %.12f %.12f\n", x[i], y[i], z[i]);
}
'''

check("the fixed checker is used for 112", J._helper_source(Path(P), "chk.cc") == J.CHECKER_FIXES / "112" / "chk.cc")
fixed = {name: J.judge_full(P, code) for name, code in (("inf", FAKE_INF), ("nan", FAKE_NAN), ("real", REAL))}
J.CHECKER_FIXES = Path("/nonexistent")   # the official checker from here on
official = {name: J.judge_full(P, code) for name, code in (("inf", FAKE_INF), ("real", REAL))}

check("official checker: the infinite claim scores full marks (the bug is real)", official["inf"]["score"] == 1.0,
      official["inf"]["score"])
check("fixed checker: the infinite claim scores 0", fixed["inf"]["status"] == "done" and fixed["inf"]["score"] == 0.0,
      (fixed["inf"]["score"], fixed["inf"]["case_info"][:1]))
check("fixed checker: a nan coordinate scores 0", fixed["nan"]["status"] == "done" and fixed["nan"]["score"] == 0.0,
      (fixed["nan"]["score"], fixed["nan"]["case_info"][:1]))
check("a real program: same score in every case under both checkers",
      official["real"]["score"] > 0 and fixed["real"]["cases"] == official["real"]["cases"],
      (official["real"]["score"], fixed["real"]["score"]))
done()
