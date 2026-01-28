# Contributing in the PACMAN Project

Before contributing, please review the following items:

1. You need to agree with the contributor license agreement detailed below.

2. We use the LLVM coding style, provided in the `.clang-format` file. We use `clang-format` 21.

    Naming convention used:
    - Function names are PascalCase.
    - Function arguments are camelCase.
    - Stack variables are lower snake_case.
    - Macros and consts are preferred to be upper SNAKE_CASE.

3. Please read the contributing rules:
   - All changes are integrated by Safran.
   - API must be kept simple and flexible.
   - It is preferred that each file serve one purpose only.
   - The code must be kept portable across all Kokkos backends. You may not use preprocessing statements to provide portability.
   - You may not use raw heap pointers. Prefer using references to objects or C++ smart pointers.
   - Please provide optional unitary and mandatory functional tests in the `tests/` folder.
   - Tests must be local (as little includes/imports as possible). Please limit the use of functions from other file and use small and simple data.
   - Favor includes (C++) and imports (Python) at the beginning of files.
   - Coverage must be kept at +95%.

4. Reporting Issues

    If you encounter any problem with the PACMAN project or have found a bug, lease check the [existing issues](www.google.com) to see if it has already been reported. If not, feel free to [open a new issue](www.google.com) with the following information:

    - A clear and concise description of the issue.
    - Steps to reproduce the issue.
    - The expected behavior.
    - The version of PACMAN you are using.
    - The Kokkos backend you activated, and the one you are using.
    - If you need to post a stacktrace, which can be verbose, please format it using markdown capabilities.
    - Any relevant test data or screenshots.

5. Feature requests

    If you have an idea for a new feature or enhancement, please [open a new issue](www.google.com) and provide the following details:

    - A clear and concise description of the feature
    - Any additional context or use cases it is meant to cover

--
Contributor License Agreement

    In order to clarify the intellectual property license granted with Contributions from any person or entity, Safran must have a Contributor License Agreement ("CLA") on file that has been signed by each Contributor, indicating agreement to the license terms below. This license is for your protection as a Contributor as well as the protection of Safran; it does not change your rights to use your own Contributions for any other purpose.
    You accept and agree to the following terms and conditions for Your present and future Contributions submitted to Safran. Except for the license granted herein to Safran and recipients of software distributed by Safran, You reserve all right, title, and interest in and to Your Contributions.

    1. Definitions.
    "You" (or "Your") shall mean the copyright owner or legal entity authorized by the copyright owner that is making this Agreement with Safran. For legal entities, the entity making a Contribution and all other entities that control, are controlled by, or are under common control with that entity are considered to be a single Contributor. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.
    "Contribution" shall mean any original work of authorship, including any modifications or additions to an existing work, that is intentionally submitted by You to Safran for inclusion in, or documentation of, any of the products owned or managed by Safran (the "Work"). For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to Safran or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, Safran for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by You as "Not a Contribution."

    2. Grant of Copyright License. Subject to the terms and conditions of this Agreement, You hereby grant to Safran and to recipients of software distributed by Safran a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute Your Contributions and such derivative works.

    3. Grant of Patent License. Subject to the terms and conditions of this Agreement, You hereby grant to Safran and to recipients of software distributed by Safran a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by You that are necessarily infringed by Your Contribution(s) alone or by combination of Your Contribution(s) with the Work to which such Contribution(s) was submitted. If any entity institutes patent litigation against You or any other entity (including a cross-claim or counterclaim in a lawsuit) alleging that your Contribution, or the Work to which you have contributed, constitutes direct or contributory patent infringement, then any patent licenses granted to that entity under this Agreement for that Contribution or Work shall terminate as of the date such litigation is filed.

    4. You represent that you are legally entitled to grant the above license. If your employer(s) has rights to intellectual property that you create that includes your Contributions, you represent that you have received permission to make Contributions on behalf of that employer, that your employer has waived such rights for your Contributions to Safran, or that your employer has executed a separate Corporate CLA with Safran.

    5. You represent that each of Your Contributions is Your original creation (see section 7 for submissions on behalf of others). You represent that Your Contribution submissions include complete details of any third-party license or other restriction (including, but not limited to, related patents and trademarks) of which you are personally aware and which are associated with any part of Your Contributions.

    6. You are not expected to provide support for Your Contributions, except to the extent You desire to provide support. You may provide support for free, for a fee, or not at all. Unless required by applicable law or agreed to in writing, You provide Your Contributions on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON- INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.

    7. Should You wish to submit work that is not Your original creation, You may submit it to Safran separately from any Contribution, identifying the complete details of its source and of any license or other restriction (including, but not limited to, related patents, trademarks, and license agreements) of which you are personally aware, and conspicuously marking the work as "Submitted on behalf of a third-party: [named here]".

    8. You agree to notify Safran of any facts or circumstances of which you become aware that would make these representations inaccurate in any respect.

---
Thank you for considering contributing to PACMAN Project! Your help is greatly appreciated.
