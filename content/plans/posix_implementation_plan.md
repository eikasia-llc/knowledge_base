# Implementation Plan - Python CLI Improvements
- status: active
- type: plan
<!-- content -->
Improve all Python programs in `manager` and `language` directories to have a POSIX-friendly command-line interface using `argparse`.

## Goal Description
- status: active
- type: task
<!-- content -->
- Standardize CLI across the codebase using `argparse`.
- Implement POSIX-friendly argument parsing (handling `-oFILE` and `-o FILE`).
- Add `-h`/`--help` with clear descriptions to all tools.
- For "single-file update" tools (e.g., `migrate.py`, `importer.py`), implement specific flexible input/output logic:
    - 1 argument: Update in-place (or default output derivation).
    - 2 arguments: Input file -> Output file (fail if target exists unless `--force`).
    - 0 arguments: Require `-i`/`--input` and `-o`/`--output`.
    - `-i` and `-o` explicit options support.

## Proposed Changes
- status: active
- type: task
<!-- content -->

### Shared Utilities
- status: active
- type: task
<!-- content -->

#### [NEW] `language/cli_utils.py`
- status: active
- type: task
<!-- content -->
- Create a shared module to handle the common argument parsing patterns.
- Implement a custom `argparse.Action` or helper to handle:
    - `-I/--in-line` flag (mutually exclusive with `-o`).
    - `-i/--input` and `-o/--output` with `action='append'` to support multiple pairs.
    - Validation logic:
        - If `-I` is present, positional arguments are treated as input files for in-place/default processing. `-i` and `-o` are forbidden.
        - If `-I` is NOT present:
             - Positional arguments are FORBIDDEN (to avoid ambiguity).
             - `-i` and `-o` must be provided.
             - `len(input_list) == len(output_list)` must hold.
             - If only 2 arguments are provided to the script (e.g. `script input output`) and they are positional *and* we want to support the strict "1 input 1 output" mode effectively, we might parse 2 positional args as in=1 out=2 IF `-i/-o/-I` are missing? **Correction based on request**: "Optionless usage is ambiguous and thus disallowed". So `script input output` matches "optionless usage". Thus, strictly require `-I` or `-i/-o`.
             - **Wait**, the user said: "It to also accepts taking two file arguments... When used in this way the first file will be an input... and the second argument will be the updated version".
             - **AND** "optionless usage is ambiguous and thus disallowed" (in the context of "python programs that handle multiple input files").
             - **Re-reading carefully**: "Improve all python programs that takes a single file as an argument... It to also accepts taking two file arguments". This refers to single-file tools.
             - "For python programs that handle multiple input files... a new parameterless option is introduced: -I/--in-line... either this option or the input/output options are mandatory."
             - **Interpretation**:
                - Tools that currently take 1 file (e.g. `md_parser.py`? No, that prints to stdout. `migrate.py` takes N files. `importer.py` takes N files.)
                - The user distinguishes between "programs that takes a single file" and "programs that handle multiple input files".
                - `migrate.py` and `importer.py` currently handle multiple files. So they fall under the "multiple input files" rule: `-I` required for batch in-place, OR `-i/-o` pairs. Optionless `migrate.py file1 file2` is DISALLOWED.
                - What about "single file" programs? The user had a requirement "accepts taking two file arguments". Which programs are these?
                  - `clean_repo.py` takes a URL.
                  - `update_master_plan.py` takes a repo URL or --all.
                  - `md_parser.py` takes 1 file (prints to stdout).
                  - `visualization.py` takes 1 file (prints to stdout).
                  - `visualize_dag.py` takes 1 file.
                  - `operations.py` takes target, source, etc.
                - It seems generic "single file" tools like `md_parser` should support `tool input output` (2 args).
                - "Multiple file" tools like `migrate.py` need `-I` or `-i/-o`.
                - I will implement the "Smart I/O" helper to support both patterns or have two variants.

### Manager Tools
- status: active
- type: task
<!-- content -->

#### [MODIFY] [clean_repo.py](file:///home/zeta/src/eikasia/central_planner/manager/clean_repo.py)
- status: active
- type: task
<!-- content -->
- Update `argparse` description.
- Ensure `-h/--help` is present.
- (No `--force` flag as it has no output option).

#### [MODIFY] [update_master_plan.py](file:///home/zeta/src/eikasia/central_planner/manager/update_master_plan.py)
- status: active
- type: task
<!-- content -->
- Refine `argparse`.

### Language Tools
- status: active
- type: task
<!-- content -->

#### [MODIFY] [md_parser.py](file:///home/zeta/src/eikasia/central_planner/language/md_parser.py)
- status: active
- type: task
<!-- content -->
- Single-file tool.
- Support `md_parser input` (print to stdout).
- Support `md_parser input output` (write to file).
- Support `-i input -o output`.
- Support `-h`.

#### [MODIFY] [visualization.py](file:///home/zeta/src/eikasia/central_planner/language/visualization.py)
- status: active
- type: task
<!-- content -->
- Single-file tool. Same as `md_parser`.

#### [MODIFY] [operations.py](file:///home/zeta/src/eikasia/central_planner/language/operations.py)
- status: active
- type: task
<!-- content -->
- Already takes named args `merge target source node`.
- Will improve with `argparse` validation and `-h`.

#### [MODIFY] [migrate.py](file:///home/zeta/src/eikasia/central_planner/language/migrate.py)
- status: active
- type: task
<!-- content -->
- Multi-file tool.
- Disallow `migrate.py file.md`.
- Require `migrate.py -I file.md` (Update in place).
- Require `migrate.py -I file1.md file2.md`.
- Support `migrate.py -i input.md -o output.md`.
- Support `migrate.py -i in1 -o out1 -i in2 -o out2`.

#### [MODIFY] [importer.py](file:///home/zeta/src/eikasia/central_planner/language/importer.py)
- status: active
- type: task
<!-- content -->
- Multi-file tool. Same as `migrate.py`.

## Verification Plan
- status: active
- type: task
<!-- content -->

### Automated Tests
- status: active
- type: task
<!-- content -->
- I will create a new test script `language/test_cli.py` to test the CLI invocation and argument logic using `subprocess` or by mocking `sys.argv` and calling `main`.
- I will verify the "fail if target exists" and "--force" behavior.

### Manual Verification
- status: active
- type: task
<!-- content -->
- Run help commands: `python language/migrate.py --help`
- Run update: `python language/migrate.py test.md`
- Run conversion: `python language/migrate.py input.md output.md`
- Run with flags: `python language/migrate.py -i input.md -o output.md`
