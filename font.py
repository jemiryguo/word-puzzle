from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
script_path = os.path.abspath(__file__)
TTF_PATH = os.path.join(os.path.dirname(script_path), "方正像素12.TTF")
TOPK = 20


def get_common_hanzi():
    hanzi = []
    for area in range(16, 88):
        for spot in range(1, 95):
            try:
                char = bytes([area + 0xA0, spot + 0xA0]).decode('gb2312')
                if '\u4e00' <= char <= '\u9fff':
                    hanzi.append(char)
            except Exception:
                continue
    return hanzi


def render_char_to_12x12_grid(char, grid, font_path, font_size=12):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"❌ 无法加载字体: {font_path}")
        return None

    img = Image.new("L", (12, 12), color=255)
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(char)
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x_offset = (12 - width) // 2 - bbox[0]
    y_offset = (12 - height) // 2 - bbox[1]

    draw.text((x_offset, y_offset), char, font=font, fill=0)

    for y in range(12):
        for x in range(12):
            pixel = img.getpixel((x, y))
            grid[y, x] = pixel < 128
    return grid


def main():
    if not os.path.exists(TTF_PATH):
        print(f"❌ 找不到 TTF 文件: {TTF_PATH}")
        return

    chars = get_common_hanzi()
    nchar = len(chars)

    data = np.zeros((nchar, 12, 12), dtype=bool)
    for i in range(nchar):
        render_char_to_12x12_grid(chars[i], data[i], TTF_PATH, 12)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    grid_display = np.ones((12, 12), dtype=float) * 0.8

    im = ax.imshow(grid_display, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))
    ax.set_xticks(np.arange(12) - 0.5, minor=True)
    ax.set_yticks(np.arange(12) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    must_be_true = set()
    must_be_false = set()

    result_text = ax.text(-0.5, -0.6, "", fontsize=10,
                          verticalalignment='bottom')

    def update_display():
        display = np.full((12, 12), 0.8)
        for (r, c) in must_be_true:
            display[r, c] = 0.0
        for (r, c) in must_be_false:
            display[r, c] = 1.0

        current_mask = np.ones(nchar, dtype=bool)
        if must_be_true:
            r_t, c_t = zip(*must_be_true)
            current_mask &= np.all(data[:, r_t, c_t], axis=1)
        if must_be_false:
            r_f, c_f = zip(*must_be_false)
            current_mask &= np.all(~data[:, r_f, c_f], axis=1)

        matching_indices = np.where(current_mask)[0]
        total = len(matching_indices)

        if total == 0:
            info = "无匹配项"
        else:
            top = [chars[i] for i in matching_indices[:TOPK]]
            if len(matching_indices) > TOPK:
                info = f"匹配({total}): [{str(top)[1:-1]}, ...]"
            else:
                info = f"匹配({total}): {top}"

        result_text.set_text(info)

        for txt in getattr(ax, '_dot_texts', []):
            txt.remove()
        ax._dot_texts = []

        if total > 0:
            first_char_idx = matching_indices[0]
            char_grid = data[first_char_idx]
            for r in range(12):
                for c in range(12):
                    if char_grid[r, c]:
                        txt = ax.text(c, r, "●",
                                      ha='center', va='center',
                                      fontsize=20,
                                      color='blue',
                                      fontweight='bold')
                        ax._dot_texts.append(txt)

        best_pos = None
        best_score = float('inf')
        best_n = (0, 0)
        if total > 1:
            candidate_data = data[current_mask]
            for r in range(12):
                for c in range(12):
                    if (r, c) in must_be_true or (r, c) in must_be_false:
                        continue
                    true_count = np.sum(candidate_data[:, r, c])
                    false_count = total - true_count
                    if true_count == 0 or false_count == 0:
                        score = total
                    else:
                        score = max(true_count, false_count)
                    if score < best_score:
                        best_score = score
                        best_pos = (r, c)
                        best_n = (true_count, false_count)

        for txt in getattr(ax, '_suggestion_texts', []):
            txt.remove()
        ax._suggestion_texts = []

        if best_pos is not None and total > 1:
            r, c = best_pos
            txt = ax.text(c, r, f'T{best_n[0]}\nF{best_n[1]}', ha='center', va='center',
                          fontsize=12, color='red', fontweight='bold')
            ax._suggestion_texts.append(txt)

        im.set_data(display)
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            if event.dblclick and event.button == 3:
                must_be_true.clear()
                must_be_false.clear()
                update_display()
            return
        col = int(np.clip(np.floor(event.xdata + 0.5), 0, 11))
        row = int(np.clip(np.floor(event.ydata + 0.5), 0, 11))
        pos = (row, col)

        if event.button == 1:  # 左键
            if pos in must_be_false:
                must_be_false.remove(pos)
            if pos in must_be_true:
                must_be_true.remove(pos)
            else:
                must_be_true.add(pos)
        elif event.button == 3:  # 右键
            if pos in must_be_true:
                must_be_true.remove(pos)
            if pos in must_be_false:
                must_be_false.remove(pos)
            else:
                must_be_false.add(pos)

        update_display()

    def on_key(event):
        if event.key == 'enter':
            print("\n✅ Final selection confirmed.")
            print(f"Must be TRUE:  {sorted(must_be_true)}")
            print(f"Must be FALSE: {sorted(must_be_false)}")

            mask = np.ones(nchar, dtype=bool)
            if must_be_true:
                r, c = zip(*must_be_true)
                mask &= np.all(data[:, r, c], axis=1)
            if must_be_false:
                r, c = zip(*must_be_false)
                mask &= np.all(~data[:, r, c], axis=1)

            indices = np.where(mask)[0]
            print(f"匹配数: {len(indices)}")
            if len(indices) > 0:
                print(f"匹配的字: {[chars[i] for i in indices]}")
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_display()
    plt.show()


if __name__ == "__main__":
    main()
