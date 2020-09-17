# -*- coding: utf-8 -*-
"""
===============================================
crypto module
===============================================

========== ====================================
========== ====================================
 Module     crypto module
 Date       2020-03-26
 Author     heewinkim
========== ====================================

*Abstract*

        *
===============================================
"""


import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES 


class _AES256():

    def __init__(self, key,block_size=32):
        self.bs = block_size
        self.key = hashlib.sha256(_AES256.str_to_bytes(key)).digest()

    @staticmethod
    def str_to_bytes(data):
        u_type = type(b''.decode('utf8'))
        if isinstance(data, u_type):
            return data.encode('utf8')
        return data

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * _AES256.str_to_bytes(chr(self.bs - len(s) % self.bs))

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]

    def encrypt(self, raw):
        raw = self._pad(_AES256.str_to_bytes(raw))
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw)).decode('utf-8')

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')


class PyCrypto(object):

    AES256 = _AES256


if __name__ == '__main__':

    crypto_obj = PyCrypto.AES256(key='key',block_size=32)

    encrypt_data = crypto_obj.encrypt('example_data')
    print(encrypt_data)  # cqw06setVz83Sy4aMpOjFeqbOKNfmRFOaIVqtYCogvFyXAhzbPrnoY+khmUfn+Q4
    decrypt_data = crypto_obj.decrypt(encrypt_data)
    print(decrypt_data)  # example_data