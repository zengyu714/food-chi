import smtplib
import socket
from email.mime.text import MIMEText
from utils.config import CONF

receiver = "zykimmy714@gmail.com"
sender = "zykimmy@126.com"

mail_user = CONF.mail_user
mail_pass = CONF.mail_pass
mail_host = "smtp.126.com"
mail_postfix = "126.com"


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def send_msg(message):
    ip = get_host_ip()
    msg = MIMEText(message, 'plain', 'utf-8')
    msg['Subject'] = f"[{ip}]: {message}"
    msg['From'] = f"{sender}"
    msg['To'] = receiver
    try:
        # use SSL send mails via 465 port
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        # login verification
        smtpObj.login(mail_user, mail_pass)
        # send
        smtpObj.sendmail(sender, receiver, msg.as_string())
        return True
    except smtplib.SMTPException as e:
        print(e)
        return False


if __name__ == '__main__':
    print(send_msg("hello"))
