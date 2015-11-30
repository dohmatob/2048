import sys
import tty
import termios


class GetArrow(object):
    def _getarrow(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            arrow = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return arrow

    def __call__(self):
        while True:
            arrow = self._getarrow()
            if arrow == "":
                return None
            while True:
                if arrow != '':
                    break
            if arrow == '\x1b[A':
                return "up"
            elif arrow == '\x1b[B':
                return "down"
            elif arrow == '\x1b[C':
                return "right"
            elif arrow == '\x1b[D':
                return "left"
            else:
                print("Not an arrow arrow!")
                return None

if __name__ == '__main__':
    gk = GetArrow()
    for _ in range(1):
        print(gk())
