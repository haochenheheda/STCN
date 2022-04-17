import os
import gdown

def Command(token, uid, output):
    return 'curl -H "Authorization: Bearer {}" https://www.googleapis.com/drive/v3/files/{}?alt=media -o {}'.format(token,uid,output)
token = 'ya29.A0ARrdaM-yQnXaj8HFslo8WOafFie5QIezennC7erbCDoQqKcJQmHbBREv77FTg5dk7kiWv-yUh06mmqb96IfEOk6gY6ZHKsL3Ht7BXmkrVOrbWNAa2K0CTEAD2S5uHAfKvLiud-vb-rEvvdv6NDW6dRxwfYFO'

os.makedirs('saves', exist_ok=True)
print('Downloading stcn.pth...')
#gdown.download('https://drive.google.com/uc?id=1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm', output='saves/stcn.pth', quiet=False)
os.system(Command(token, '1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm', 'saves/stcn.pth'))
print('Done.')
