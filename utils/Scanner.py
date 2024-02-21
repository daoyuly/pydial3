###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2019
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

'''
Scanner.py - string scanning utility 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    none

************************

'''

import io
import tokenize


def remove_comments(src):
    """
    This reads tokens using tokenize.generate_tokens and recombines them
    using tokenize.untokenize, and skipping comment/docstring tokens in between
    """
    f = io.StringIO(src)
    processed_tokens = []
    last_token = None
    # go thru all the tokens and try to skip comments and docstrings
    for tok in tokenize.generate_tokens(f.readline):
        t_type, t_string, t_srow_scol, t_erow_ecol, t_line = tok

        if t_type == tokenize.COMMENT:
            pass

        # elif t_type == tokenize.STRING:
        #     if last_token is None or last_token[0] in [tokenize.INDENT]:
        #         pass

        else:
            processed_tokens.append(tok)

        last_token = tok

    tokens = []
    prev = tokenize.NL
    for t in processed_tokens:
        if t[0] != prev or t[0] in [tokenize.ERRORTOKEN, tokenize.STRING, tokenize.OP, tokenize.NAME, tokenize.NUMBER]:
            tokens.append(t)
        prev = t[0]
    final_list = ''.join([t.string for t in tokens if t[0] != tokenize.NL])
    return final_list


class Scanner(object):
    '''
    Class to maintain tokenized string.
    '''
    def __init__(self, string):
        src = io.StringIO(string).readline
        self.tokens = tokenize.generate_tokens(src)
        self.cur = None

    def __next__(self):
        while True:
            self.cur = next(self.tokens)
            if self.cur[0] not in [tokenize.NEWLINE] and self.cur[1] != ' ':
                break
        return self.cur

    def check_token(self, token, message):
        if type(token) == int:
            if self.cur[0] != token:
                raise SyntaxError(message + '; token: %s' % str(self.cur))
        else:
            if self.cur[1] != token:
                raise SyntaxError(message + '; token: %s' % str(self.cur))
