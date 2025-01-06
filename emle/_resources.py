#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2025
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# EMLE-Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EMLE-Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""Resource handling utilities for EMLE."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"


def _fetch_resources():
    """Fetch resources required for EMLE."""

    import os as _os
    import pygit2 as _pygit2
    import sys

    if _os.environ.get("EMLE_NO_RESOURCE_CHECK") is not None:
        return

    # Create the name for the expected resources directory.
    resource_dir = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "resources"
    )

    # Check if the resources directory exists.
    if not _os.path.exists(resource_dir):
        try:
            # If it doesn't, clone the resources repository.
            print("Downloading EMLE resources...", file=sys.stderr)
            _pygit2.clone_repository(
                "https://github.com/chemle/emle-models.git", resource_dir
            )
        except Exception as e:
            print(f"Failed to download EMLE resources: {e}", file=sys.stderr)
    else:
        # If it does, open the repository and pull the latest changes.
        try:
            repo = _pygit2.Repository(resource_dir)
            _pull(repo)
        except Exception as e:
            print(f"Failed to update EMLE resources: {e}", file=sys.stderr)


# The MIT License (MIT)

# Copyright (c) 2015 Michael Boselowitz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def _pull(repo, remote_name="origin", branch="main"):
    """
    Pull the latest changes from the remote repository.

    Taken from:
    https://github.com/MichaelBoselowitz/pygit2-examples/blob/master/examples.py
    """

    import pygit2 as _pygit2

    for remote in repo.remotes:
        if remote.name == remote_name:
            remote.fetch()
            remote_master_id = repo.lookup_reference(
                "refs/remotes/origin/%s" % (branch)
            ).target
            merge_result, _ = repo.merge_analysis(remote_master_id)
            # Up to date, do nothing
            if merge_result & _pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
                return
            # We can just fastforward
            elif merge_result & _pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
                print("Updating EMLE resources...")
                repo.checkout_tree(repo.get(remote_master_id))
                try:
                    master_ref = repo.lookup_reference("refs/heads/%s" % (branch))
                    master_ref.set_target(remote_master_id)
                except KeyError:
                    repo.create_branch(branch, repo.get(remote_master_id))
                repo.head.set_target(remote_master_id)
            elif merge_result & _pygit2.GIT_MERGE_ANALYSIS_NORMAL:
                print("Updating EMLE resources...")
                repo.merge(remote_master_id)

                if repo.index.conflicts is not None:
                    for conflict in repo.index.conflicts:
                        print("Conflicts found in:", conflict[0].path)
                    raise AssertionError("Conflicts!")

                user = repo.default_signature
                tree = repo.index.write_tree()
                commit = repo.create_commit(
                    "HEAD",
                    user,
                    user,
                    "Merge!",
                    tree,
                    [repo.head.target, remote_master_id],
                )
                # We need to do this or git CLI will think we are still merging.
                repo.state_cleanup()
            else:
                raise AssertionError("Unknown merge analysis result")
