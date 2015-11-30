# Synopsis: 2048 game implementation in Python (Python 2/3 compatible)
# Author: Elvis Dohmatob <gmdopp@gmail.com>

from __future__ import print_function
import sys
try:
    import numpy as np
except ImportError:
    print("Oops! Looks like you haven't installed the numpy library.")
    print("Bye!")
    sys.exit(1)
from _get_arrow import GetArrow

# raw_input doesn't exist in Python3 (renamed to "input")
input_grabber = input if sys.version_info.major == 3 else raw_input

banner = """
  ________________________________________________________________
< 2048 game dedicated to the young dr. alex jr., @ parietal _team_ >
  ----------------------------------------------------------------
         \   ^__^
          \  (oo)\_______
             (__)\       )\/\

                 ||----w |
                 ||     ||
"""


class Game2048(object):

    _SIZES = [2, 4, 8]  # allowed sizes for the game grid

    def __init__(self, size=None, grid=None, random_state=None, mode='arrows'):
        # misc
        if not mode in ["arrows", "letters"]:
            raise ValueError("Invalid mode: %s" % mode)
        self.mode = mode
        self.size = size
        self.grid = grid
        self.random_state = random_state
        self.load_batteries()

    def load_batteries(self):
        """Some serious conf business."""
        print(banner)
        self.rng_ = np.random.RandomState(self.random_state)

        # get size of grid
        if self.size is None:
            while not self.size in self._SIZES:
                try:
                    self.size = int(input_grabber(
                        ("<> Enter game size (can be %s):"
                         " ") % ", ".join([str(s) for s in self._SIZES])))
                except ValueError:
                    continue

        # make grid
        if self.grid is None:
            self.grid_ = np.zeros((self.size, self.size), dtype=int)
            indices = list(np.ndindex((self.size, self.size)))
            support = self.rng_.choice(range(len(indices)), 2)
            self.grid_[indices[support[0]][0], indices[support[0]][1]] = 2
            self.grid_[indices[support[0]][1],
                       indices[support[1]][1]] = self.rng_.choice([2, 4])
        else:
            self.grid_ = np.array(self.grid).copy()

    def get_params(self):
        """Get all parameters of class instance."""
        return dict(size=self.size, grid=self.grid_,
                    random_state=self.random_state)

    def get_move(self):
        """Get next move from input sensor (screen, etc.)."""
        print(self)
        if self.mode == "arrows":
            mv = GetArrow()()
            if not mv is None:
                return dict(up="I", down="K", left="J", right="L")[mv]
            return None
        mv = "X"
        while not mv in ["I", "J", "K", "L"]:
            mv = input_grabber(("<> Enter direction for movement "
                                "(I=u,J=left,K=down, L=right) : "))

            # check empty line
            if not mv:
                return
        return mv

    def clone(self):
        """Clone class instance."""
        return Game2048(*self.get_params)

    def __repr__(self):
        """Converts "2048" board to a string."""
        out = ""
        line = None
        for line in [[(str(x) if x else "").center(6) + "|" for x in line]
                     for line in self.grid_]:
            line = "|%s" % "".join(line)
            out += "%s\n" % line
            out += "%s\n" % ("-" * len(line))
        out = "%s\n%s" % ("-" * len(line), out)
        return out

    def evolve(self):
        """Fill a random empty cell with 2 (or 4, with very small proba)."""
        z = np.where(self.grid_ == 0.)
        i = self.rng_.choice(len(z[0]))
        self.grid_[z[0][i], z[1][i]] = self.rng_.choice([2, 4], p=[.9, .1])
        return self

    def _squeeze(self, v, down=True):
        """Reorganize vector into a contiguous chunk of nonzero stuff, padded
        with zeros."""
        nz = [x for x in v if x]
        if down:
            return np.append(np.zeros(len(v) - len(nz)), nz)
        else:
            return np.append(nz, np.zeros(len(v) - len(nz)))

    def _vertical_move(self, down=True):
        """Helper method for doing up and down moves."""
        places = range(self.size)
        places = places[:0:-1] if down else places[:-1]
        for j in range(self.size):
            col = self._squeeze(self.grid_[:, j], down=down)
            for i in places:
                salt = 1 - 2 * down
                if col[i] == col[i + salt]:
                    # merge cells (i, j) and (i + salt, j)
                    col[i] *= 2
                    col[i + salt] = 0
                    col = self._squeeze(col, down=down)
            self.grid_[:, j] = col
        return self

    def up(self):
        """Perform upward move."""
        return self._vertical_move(down=False)

    def down(self):
        """Perform downward move."""
        return self._vertical_move(down=True)

    def _horizontal_move(self, right=True):
        """Helper method for doing left and right moves.

        The trick is to simply do a horizontal movement on the transpose
        game."""
        params = self.get_params()
        params["grid"] = self.grid_.T
        self.grid_ = Game2048(**params)._vertical_move(down=right).grid_.T
        return self

    def left(self):
        """Perform leftward move."""
        return self._horizontal_move(right=False)

    def right(self):
        """Perform rightward move."""
        return self._horizontal_move(right=True)

    def move(self, mv):
        """Make specified move.

        I = up, J = left, K = down, L = right.
        """
        if mv == "I":
            return self.up()
        elif mv == "J":
            return self.left()
        elif mv == "K":
            return self.down()
        elif mv == "L":
            return self.right()
        else:
            raise ValueError("Invalid move: %s" % mv)

    @property
    def score(self):
        """Get running score."""
        return np.max(self.grid_)

    @property
    def full(self):
        """Check whether grid is full."""
        return np.all(self.grid_ != 0.)

    @property
    def ended(self):
        """Check whether game has ended."""
        return self.full or self.score >= 2048

    def play(self):
        """Play game."""
        if self.mode == "arrows":
            print("Use arrows (up, down, left, right) to play game")
        else:
            print(("The player enters direction for movement : I = up, J = "
                   "left, K = down, L = right"))
        print("Note : enter an empty line to stop the game")
        while True:
            mv = self.get_move()
            if mv is None:
                print("Game aborted by user")
                break
            self.move(mv)
            if self.full:
                break
            self.evolve()
        print(self)
        if self.score < 2048:
            print("Game over. Your score is %i" % self.score)
        else:
            print("Bravo! You completed the game with score %i" % self.score)


if __name__ == "__main__":
    Game2048().play()
