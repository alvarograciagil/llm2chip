# Contributing to llm2chip

Thank you for your interest. Bug reports, questions, ideas and pull requests are all welcome.

## Before you start

- **Bugs and questions:** open an issue. For a bug, say what you did, what you expected and what happened, with the model, the design parameters and your system (macOS, Linux or Windows; GPU if any).
- **Larger changes:** open an issue first to talk about the approach, especially for changes to the generated hardware, the numeric formats or the file formats. It saves work on both sides.
- **Security problems:** do not open a public issue. Use GitHub's private vulnerability reporting on the repository instead.

[Installing and running llm2chip](docs/install.md) explains how to build it.

## Making a change

1. Fork the repository and create a branch from `main`.
2. Make the change, with tests for it. The simulator runs what the compiler produces, so a change to the hardware should be checked against the arithmetic it replaces, not only for running without errors.
3. Run the checks:

   ```bash
   scripts/check.sh
   ```

   On Windows, `scripts\check.ps1`. It runs the Rust tests, Clippy with warnings as errors, the web build and the web tests. If your change touches the generated hardware (`crates/netlist`, the lowering in `crates/sim`, the firmware), also run the slow set, which builds whole chips in gates:

   ```bash
   scripts/check.sh --full
   ```

4. Open a pull request describing what changes and why, and how you checked it. Say in it that you agree to the [contributor license](#contributor-license) below.

## Conventions

- **Language.** Code, comments, documentation and commit messages are in English.
- **The interface is in English and Spanish.** Every text shown to the user goes through `t()` and needs its Spanish in `web/src/i18n.ts`, or a pattern in `web/src/engineText.ts` for text built from numbers. `web/src/i18nCoverage.test.ts` fails if a text is missing its translation. If you do not speak Spanish, say so in the pull request and it will be translated.
- **Figures say where they come from.** The estimator labels each figure as derived, calibrated or assumed. Keep new figures labelled, and give the source of calibrated ones.
- **Documentation follows the code.** If a change alters what a parameter does, how something is started or what a figure means, update `docs/` in the same pull request.
- **No weights or secrets in the repository.** Model weights are not committed; the small models in `tests/golden` are generated for the tests. Never commit tokens, keys or paths from your own machine.
- **Commits.** One topic per commit, with a short summary line and, when useful, a paragraph on why.

## Contributor license

llm2chip is released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md), and its author also offers separate commercial licenses. For that to remain possible, contributions need a license broader than the noncommercial one.

By submitting a contribution (code, documentation, data or any other material) to this project, you agree that:

1. You keep the copyright in your contribution.
2. You grant Álvaro Graciá Gil, the author of llm2chip, a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to use, reproduce, modify, prepare derivative works of, publicly display, publicly perform, sublicense and distribute your contribution and such derivative works, under any license terms, including commercial and noncommercial ones.
3. You grant the same parties a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license for any of your patent claims that your contribution necessarily infringes, alone or combined with the project.
4. Your contribution is your original work, or you have the right to submit it under these terms. If your employer has rights to what you create, you have their permission to contribute it or they have waived those rights.
5. If your contribution includes third-party material, you say so in the pull request, with its source and license.
6. You are not expected to provide support for your contribution, and it is provided "as is", without warranties of any kind.

The contribution itself is then published in this repository under the project's license, together with the rest of the code.

## Code of conduct

Be respectful and constructive. Discuss the work, not the person. Harassment and personal attacks are not tolerated, and the maintainer may remove comments, issues or contributions that cross that line.
