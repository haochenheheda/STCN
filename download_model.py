import os
import gdown
import sys

def Command(token, uid, output):
    return 'curl -H "Authorization: Bearer {}" https://www.googleapis.com/drive/v3/files/{}?alt=media -o {}'.format(token,uid,output)
token = sys.argv[1]

os.makedirs('saves', exist_ok=True)
print('Downloading stcn.pth...')
#gdown.download('https://drive.google.com/uc?id=1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm', output='saves/stcn.pth', quiet=False)
os.system(Command(token, '1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm', 'saves/stcn.pth'))
print('Done.')
