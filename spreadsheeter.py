import xlsxwriter
import os

def output(rows, cols, boxes, path):
  try:
    os.remove(path)
  except OSError:
    pass

  try:
    book = xlsxwriter.Workbook(path)
    sheet = book.add_worksheet()
  
    indices = {}
  
    for i, row in enumerate(rows):
  
      for box in row[5]:
        idx = boxes.index(box)
        indices[idx] = {}
  
        indices[idx]['row'] = i
  
    for i, col in enumerate(cols):
      for box in col[5]:
        idx = boxes.index(box)
        indices[idx]['col'] = i

    cells = [[[] for x in range(len(cols))] for y in range(len(rows))]

    if path.endswith('2009-01-28_12_71-74.jpg.xlsx'):
      import pdb;pdb.set_trace()
      print('what')

    for i,box in enumerate(boxes):
      cells[indices[i]['row']][indices[i]['col']].append(box)

    for row_idx, row in enumerate(cells):
      for col_idx, cell in enumerate(row):
        sorted_labels = [box[4] for box in sorted(cell, key = lambda x: x[0])]
        flat_labels = []

        for labels in sorted_labels:
          flat_labels += labels

        sheet.write(row_idx, col_idx, ' '.join(flat_labels))

    #
  
    # sheet.write(row, col, item)

  finally:
    book.close()
