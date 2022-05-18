import os
import zipfile
import sys


def Command(token, uid, output):
    return 'curl -H "Authorization: Bearer {}" https://www.googleapis.com/drive/v3/files/{}?alt=media -o {}'.format(token,uid,output)

token = sys.argv[1]
uid = sys.argv[2]
output = sys.argv[3]


os.system(Command(token, uid, output))
