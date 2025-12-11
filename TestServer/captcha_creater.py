from random import randint, choices
import string
from captcha import image
import secrets
from PIL import Image
from PIL.ImageDraw import Draw
from PIL.ImageFilter import SMOOTH
from PIL.Image import new as createImage

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
    def __init__(self):
        super().__init__()

    @staticmethod
    def my_random_color():
        return tuple([randint(0, 255) for _ in range(3)])

    @staticmethod
    def create_noise_curve(image: Image, color, lines:int=1) -> Image:
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

def generate_captcha(path):
    """ 生成验证码 """
    code = ''.join(choices((string.ascii_letters + string.digits), k=4))
    name = ''.join(choices((string.ascii_letters + string.digits), k=20))
    My_CAPTCHA().write(code, f'{path}/{name}.png', lines=randint(1, 4))
    return name, code