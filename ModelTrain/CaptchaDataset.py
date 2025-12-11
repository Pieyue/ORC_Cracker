import os
import string
import secrets
import pandas as pd
import numpy as np
from captcha import image
from random import choices, randint
from PIL import Image
from PIL.ImageDraw import Draw
from PIL.ImageFilter import SMOOTH
from PIL.Image import new as createImage
import tensorflow as tf

__all__ = ['Dataset', 'My_CAPTCHA']

def random_color(
        start: int,
        end: int,
        opacity: int | None = None):
    red = secrets.randbelow(end - start + 1) + start
    green = secrets.randbelow(end - start + 1) + start
    blue = secrets.randbelow(end - start + 1) + start
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity

class My_CAPTCHA(image.ImageCaptcha):
    """
    修改方法：
    create_noise_curve()：   随机数量、粗细彩色干扰线
    create_noise_dots()：    随机数量、大小彩色噪点
    create_noise_dots()：    使每一个字符颜色不同
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def my_random_color():
        return tuple([randint(0, 255) for _ in range(3)])

    @staticmethod
    def create_noise_curve(image: Image, color, lines:int=1) -> Image:
        """ 修改干扰线数量 """
        for _ in range(lines):
            color = My_CAPTCHA.my_random_color()
            w, h = image.size
            x1 = secrets.randbelow(int(w / 5) + 1)
            x2 = secrets.randbelow(w - int(w / 5) + 1) + int(w / 5)
            y1 = secrets.randbelow(h - 2 * int(h / 5) + 1) + int(h / 5)
            y2 = secrets.randbelow(h - y1 - int(h / 5) + 1) + y1
            points = [x1, y1, x2, y2]
            end = secrets.randbelow(41) + 160
            start = secrets.randbelow(21)
            Draw(image).arc(points, start, end, fill=color, width=randint(1, 3))
        return image

    @staticmethod
    def create_noise_dots(
            image: Image,
            color,
            width: int = 3,
            number: int = 30) -> Image:
        """ 修改噪点生成数量和颜色 """
        draw = Draw(image)
        w, h = image.size
        number = randint(30, 50)
        while number:
            width = randint(3, 5)
            color = My_CAPTCHA.my_random_color()
            x1 = secrets.randbelow(w + 1)
            y1 = secrets.randbelow(h + 1)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(
            self,
            chars: str,
            color,
            background) -> Image:
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = createImage('RGB', (self._width, self._height), background)
        draw = Draw(image)

        images: list[Image] = []
        for c in chars:
            """ 修改每个验证码字符的颜色 """
            color = My_CAPTCHA.my_random_color()
            if secrets.randbits(32) / (2**32) > self.word_space_probability:
                images.append(self._draw_character(" ", draw, color))
            images.append(self._draw_character(c, draw, color))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(self.word_offset_dx * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(self.lookup_table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + (-secrets.randbelow(rand + 1))

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars: str, bg_color = None, fg_color = None, lines:int=1) -> Image:
        """Generate the image of the given characters.

        :param chars: text to be generated.
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        background = bg_color if bg_color else random_color(238, 255)
        random_fg_color = random_color(10, 200, secrets.randbelow(36) + 220)
        color = fg_color if fg_color else random_fg_color

        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color, lines=lines)
        im = im.filter(SMOOTH)
        return im

    def write(self, chars: str, output: str, format: str = 'png',
              bg_color= None,
              fg_color= None,
              lines:int=1) -> None:
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        im = self.generate_image(chars, bg_color=bg_color, fg_color=fg_color, lines=lines)
        im.save(output, format=format)


class Dataset:
    """ 创建数据集和预处理验证码图像 """
    def __init__(self, height=60, width=160, length=4, train=80000, test=10000, validation=10000):
        """
        :param height:      图像高度
        :param width:       图像宽度
        :param train:       训练集大小
        :param test:        测试集大小
        :param validation:  验证集大小
        """
        self.target = (height, width)
        self.length = length
        self.train = train
        self.test = test
        self.validation = validation

    def build_dataset(self):
        """ 创建数据集 """
        root = os.getcwd()
        datasets_dir = os.path.join(root, 'datasets')

        train_dir = os.path.join(datasets_dir, 'train')
        test_dir = os.path.join(datasets_dir, 'test')
        validation_dir = os.path.join(datasets_dir, 'val')

        train_images_dir = os.path.join(train_dir, 'images')
        test_images_dir = os.path.join(test_dir, 'images')
        validation_images_dir = os.path.join(validation_dir, 'images')
        if not os.path.exists(train_images_dir):
            os.makedirs(train_images_dir)
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
        if not os.path.exists(validation_images_dir):
            os.makedirs(validation_images_dir)

        print("开始创建数据集……")
        print('开始创建训练集……')
        with open(f'{train_dir}/labels.csv', 'w') as f:
            for i in range(self.train):
                self.generate_captcha(train_images_dir, f, i)
        print('训练集创建完成')

        print('开始创建测试集……')
        with open(f'{test_dir}/labels.csv', 'w') as f:
            for i in range(self.test):
                self.generate_captcha(test_images_dir, f, i)
        print('测试集创建完成')

        print("开始创建验证集……")
        with open(f'{validation_dir}/labels.csv', 'w') as f:
            for i in range(self.validation):
                self.generate_captcha(validation_images_dir, f, i)
        print('验证集创建完成')
        print("数据集创建完成")

    def generate_captcha(self, path, file, num: int | None=0):
        """
        生成验证码
        :param path:    输出位置
        :param file:    标签文件对象
        :param num:     验证码长度
        """
        code = ''.join(choices((string.ascii_letters + string.digits), k=4))
        lines = randint(1, 4)
        if num is not None:
            My_CAPTCHA().write(code, f'{path}/{code}_{num}.png', lines=lines)
        else:
            My_CAPTCHA().write(code, f'{code}.png', lines=lines)
            print('Code：', code)
        if file:
            file.write(f'images/{code}_{num}.png,{code}\n')

    def create_one_captcha(self, path='./'):
        """
        生成单张图像
        :param path: 保存位置
        """
        self.generate_captcha(path=path, file=None, num=None)

    def load_labels(self, file_path):
        """
        加载labels.csv文件
        :param file_path:   csv文件位置
        :return:            (每张图像位置，验证码)
        """
        df: pd.DataFrame = pd.read_csv(file_path)
        return df.iloc[:, 0].values, df.iloc[:, 1].values

    def captcha_image_generator(self, class_dir, name_list, label_list, channels=3):
        """
        数据管道，将验证码图片和文本标签转换为浮点数图像张量和 one-hot 编码标签
        :param class_dir:   train/validation 目录
        :param name_list:   验证码列表
        :param label_list:  验证码标签列表
        :param channels:    通道数
        :return:            tf.data.Dataset对象
        """
        def load_and_preprocess(img_name, label):
            """
            加载对应的图像并预处理
            :param img_name:    单个在磁盘中的图像的文件名
            :param label:       对应的标签
            :return:            (张量类型的图片,标签映射表)
            """
            # 将图像路径转换为tensor
            img_path = tf.strings.join([class_dir, '/', img_name])

            # 使用TensorFlow原生的图像加载和预处理
            image = tf.io.read_file(img_path)
            image = tf.image.decode_png(image, channels=channels)
            image = tf.image.convert_image_dtype(image, tf.float32)  # uint8->float32，自动标准化到[0,1]
            image = tf.image.resize(image, self.target)

            # 编码标签 - 使用统一的TensorFlow方法，constant将字符串转为张量
            char_set = tf.constant(string.digits + string.ascii_letters)

            # 将字符串拆分为字符，类似于list()
            chars = tf.strings.bytes_split(label)

            # 将字符映射到索引
            table = tf.lookup.StaticHashTable(                                          # 创建一个静态哈希表，用于键值对查找
                tf.lookup.KeyValueTensorInitializer(                                    # 初始化键值对
                    tf.strings.unicode_split(char_set, 'UTF-8'),          # 将字符集按UTF-8编码拆分，类似于list()
                    tf.range(tf.strings.length(char_set), dtype=tf.int64)               # 生成从0开始的，len(char_set)长度的连续整数
                ),
                default_value=-1    # 查找不存在的键时返回-1
            )

            indices = table.lookup(chars)                                               # 将chars映射为对应的索引数组
            encoded_label = tf.one_hot(indices, depth=62)                               # 将索引转换为one-hot,62为所有字符类别
            encoded_label = tf.reshape(encoded_label, (4, 62))                    # 将one-hot结构转为二维张量

            return image, encoded_label

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((name_list, label_list))  # 创建数据集对象，(图像名称,标签)

        # 对每一对name, label应用load_and_preprocess
        # num_parallel_calls指定在处理时调用多少个线程，tf.data.AUTOTUNE自动选择
        return dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)