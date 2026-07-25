from src.pipeline_a.layout import group_into_lines, line_text
from src.pipeline_a.ocr import OCRBox


def box(text, x1, y1, x2, y2):
    return OCRBox(text=text, score=1.0, x1=x1, y1=y1, x2=x2, y2=y2)


class TestGroupIntoLines:
    def test_empty(self):
        assert group_into_lines([]) == []

    def test_single_box(self):
        lines = group_into_lines([box("A", 0, 0, 10, 20)])
        assert len(lines) == 1
        assert [b.text for b in lines[0]] == ["A"]

    def test_same_row_different_x_grouped_together(self):
        # 同一列的三個欄位（名稱/數量/金額），y 幾乎相同但 x 分散
        boxes = [
            box("金額", 300, 389, 350, 412),
            box("品名", 106, 386, 152, 413),
            box("數量", 247, 387, 413 - 26, 413),
        ]
        lines = group_into_lines(boxes)
        assert len(lines) == 1
        # 須依 x 排序回左到右
        assert [b.text for b in lines[0]] == ["品名", "數量", "金額"]

    def test_distinct_rows_kept_separate(self):
        boxes = [
            box("第一行", 100, 100, 200, 130),
            box("第二行", 100, 200, 200, 230),
            box("第三行", 100, 300, 200, 330),
        ]
        lines = group_into_lines(boxes)
        assert len(lines) == 3
        assert [line_text(l) for l in lines] == ["第一行", "第二行", "第三行"]

    def test_rotation_scrambles_row_grouping(self):
        """模擬旋轉後同一列的 y 座標隨 x 飄移——這正是 deskew 存在的理由。"""
        boxes = [
            box("品名", 100, 400, 150, 430),
            box("數量", 250, 410, 300, 440),  # y 因傾斜而偏移
            box("金額", 350, 420, 400, 450),  # 偏移更多
        ]
        lines = group_into_lines(boxes, y_tolerance_ratio=0.3)
        # 容忍度不足以把它們視為同一列，驗證「未校正旋轉會拆散欄位」的假設
        assert len(lines) > 1


class TestLineText:
    def test_concatenates_without_separator(self):
        line = [box("拿鐵咖啡", 0, 0, 50, 20), box("60", 60, 0, 80, 20)]
        assert line_text(line) == "拿鐵咖啡60"
