# Development notes

A historical snapshot, kept because the process is part of what this repository
documents. These files were written *during* the build, not after it, and some
of their figures are superseded — `docs/RESULTS.md` and `docs/metrics.json` are
the authority for any number.

| | |
|---|---|
| `AGENTS.md` | how the work was split across specialised agents |
| `HANDOVER.md` | the state of the tree at handover, written for a fresh reader |
| `copilot_dotgithub/` | the GitHub Copilot configuration: agent definitions, path-scoped instructions and task prompts. Copy to `.github/` to use it. |

The most transferable idea in here is `copilot_dotgithub/agents/tsr-kernel.agent.md`:
an agent whose stated job is to refuse any kernel change that cannot be shown to
preserve bit-exactness. Having one role whose default answer is *no* is why the
speed and RAM work never quietly cost accuracy.

`copilot_dotgithub/agents/tsr-lead.agent.md` says "do not chase 364/364" and
refers to *both* residual errors. That was true of an earlier model; the shipped
one has a single error. Left as written.
