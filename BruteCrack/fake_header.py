"""
Generate random HTTP Header.
"""

from random import choice, randint
import string
import os
WORKPATH = os.path.abspath(__file__)
WORKDIR = os.path.dirname(WORKPATH)

UA_LIST = [line.strip() for line in open(f'{WORKDIR}/config/User_Agent.conf', 'r', encoding='utf-8')]
REFERERS = [
    'https://www.baidu.com/',
    'https://cn.bing.com/',
    'https://www.google.com/',
    'https://www.souhu.com/',
    'https://www.hao123.com/',
    'https://www.so.com/',
    'https://www.sougou.com/',
    'https://www.qq.com/'
]
ACCEPT_ENCODING = ['gzip', 'deflate', 'br', 'gzip, deflate']

def random_ip():
    """
    :return:随机ip
    """
    one = choice([
        randint(11, 126),
        randint(128, 168),
        randint(170, 171),
        randint(173, 191),
        randint(193, 197),
        randint(199, 202),
        randint(204, 223)
        ])
    two = three = four = randint(0, 255)
    return f'{one}.{two}.{three}.{four}'

def random_cookie(n=16):
    chars = string.ascii_letters + string.digits
    return ''.join([choice(chars) for _ in range(n)])


def generate_headers(cookie=None):
   fake_ip = random_ip()
   Cookie = cookie if cookie else 'sessionid='+random_cookie(20)
   Accept = choice(["text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "*/*"])
   UA = choice(UA_LIST)
   Referer = choice(REFERERS)
   AC_encoding = choice(ACCEPT_ENCODING)
   Languages = choice(['zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7', 'en-US,en;q=0.9', 'ja,en-US;q=0.8,en;q=0.6', 'ko,en-US;q=0.7,en;q=0.5'])
   return {
       'Connection': 'keep-alive',
       'User-Agent': UA,
       'Accept': Accept,
       'Accept-Encoding': AC_encoding,
       'Accept-Language': Languages,
       'Referer': Referer,
       'X-Forwarded-For': fake_ip,
       'X-Real-IP': fake_ip,
       'Cookie': Cookie
   }
