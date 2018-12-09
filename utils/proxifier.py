"""
Ref: Proxy supported via: https://www.16yun.cn/help/ss_demo/#1python
"""
import requests
import random
import time
from utils.config import CONF


def request_proxifier(url):
    # https_url = "https://httpbin.org/ip"

    # set proxy host
    proxyHost = CONF.proxy_host
    proxyPort = CONF.proxy_port

    # proxy verification
    proxyUser = CONF.proxy_user
    proxyPass = CONF.proxy_pass

    proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
        "host": proxyHost,
        "port": proxyPort,
        "user": proxyUser,
        "pass": proxyPass,
    }

    # http for both http and https protocol
    proxies = {
        "http" : proxyMeta,
        "https": proxyMeta,
    }

    # switch headers
    random.seed(time.time())
    tunnel = random.randint(1, 10000)
    headers = {"Proxy-Tunnel": str(tunnel)}

    resp = requests.get(url, proxies=proxies, headers=headers)
    return resp
