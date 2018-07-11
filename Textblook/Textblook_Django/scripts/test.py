# Full path and name to your csv file 
csv_filepathname="C:\Project\TextbookProj\\tests\\sample_data_a.csv" 
# Full path to your django project directory 
your_djangoproject_home="C:\\Project\\TextbookProj\\django_proj\\textblook" 

import sys,os 
sys.path.append(your_djangoproject_home) 
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings' 

from book.models import Textbook 
import csv 
dataReader = csv.reader(open(csv_filepathname), delimiter=',', quotechar='"') 
for row in dataReader: 
	# Ignore the header row, import everything else 
	book = Textbook() 
	book.title = row[0] 
	book.author = row[1] 
	book.isbn = row[2]
	book.image = row[3]
	book.klass = row[10]
	book.save()
