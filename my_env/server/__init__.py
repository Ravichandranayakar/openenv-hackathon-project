# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env environment server components."""

# Support both in-repo and standalone imports
try:
  # In-repo imports
  from .customer_support_environment import CustomerSupportEnvironment
except ImportError:
  # Standalone imports
  from server.customer_support_environment import CustomerSupportEnvironment

__all__ = ["CustomerSupportEnvironment"]
