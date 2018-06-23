# see https://www.kaggle.com/jiweiliu/outbrain-click-prediction/extract-leak-in-30-mins-with-small-memory

import os
import csv
import zipfile

print '=== Reading promoted_content.csv.zip....'
doc_ids = set()
with zipfile.ZipFile('data/promoted_content.csv.zip') as zip:
    with zip.open('promoted_content.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_ids.add(int(row['document_id']))
print '%s promoted docs.\n' % len(doc_ids)


print '=== Reading page_views.csv.zip...'
leak = {}
page_view_file = 'page_views_sample.csv'
#page_view_file = 'page_views.csv'
with zipfile.ZipFile('data/' + page_view_file + '.zip') as zip:
    with zip.open(page_view_file) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 1000000 == 0:
                print i
            
            doc_id = int(row['document_id'])
            if doc_id not in doc_ids: continue
            if leak.has_key(doc_id):
                leak[doc_id].add(row['uuid'])
            else: leak[doc_id] = set([row['uuid']])
print 'Find %d leaked docs.\n' % len(leak)

print '=== Writing to file...'
with open('tmp/leaked_docs.csv', 'w') as f:
    f.write('document_id,uuids\n')
    for k in sorted(leak.keys()):
        v = sorted(list(leak[k]))
        f.write('%s,%s\n' % (k, ' '.join(v)))

