#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# FinTools: Portfolio analytics
# https://github.com/zegoverno/zenfin
#
# Copyright 2021 José Governo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import version

__version__ = version.version
__author__ = "José Governo"


from . import stats, utils, reports

__all__ = ['stats', 'reports', 'utils']