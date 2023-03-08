#!/usr/bin/env python
import sys


def fix_file(kitti_file: str, initial_pos_x: float, initial_pos_y: float, file_content: str):
    first_line = f'1.0 0.0 0.0 {initial_pos_x} 0.0 1.0 0.0 {initial_pos_y} 0.0 0.0 1.0 0.0'

    out_file = open(kitti_file, 'w')
    out_file.write(first_line)
    out_file.write('\n')
    out_file.write(file_content)
    out_file.close()

    pass


def main() -> None:
    if len(sys.argv) < 4:
        pass
    kitti_file = sys.argv[1]
    initial_pos_x = float(sys.argv[2])
    initial_pos_y = float(sys.argv[3])

    in_file = open(kitti_file, 'r')
    file_content = in_file.read()
    in_file.close()
    lines = file_content.split('\n')

    first_line_args = list(map(float, lines[0].split(' ')))

    print(f'{first_line_args[3]}, {first_line_args[7]}\n{initial_pos_x}, {initial_pos_y}')

    if first_line_args[3] != initial_pos_x or first_line_args[7] != initial_pos_y:
        fix_file(kitti_file, initial_pos_x, initial_pos_y, file_content)

    pass


if __name__ == "__main__":
    main()