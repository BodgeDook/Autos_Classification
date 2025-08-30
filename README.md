The description:

This project includes the database of current car models and brands which was reformed by me. The most part of the time
was spent on building the data so it could look and be workable. It uses simple knn algorithm to know the mark of
your car by the database's features. There are only few car marks but can be added more later (we'll see...).

Also, some notes about used libraries:
  1) To read the excel-data you need to import the 'openpyxl' lib, so pandas would be able to read the 'datasets' like mine.
  2) Another note is about scikit-learn: I used scale and KNeighborsClassifier to learn the n-n (scale is used to
  generalise all values to a common 'mashstab' or a value).
  3) Also, I used one-hot-encoding to encode the string parts of the data into the numbers. It belongs to the pandas lib and looks like: 'pd.get_dummies(...)'.
     I can say that this function already exists in a pandas lib, so you don't need to donwload any other libs like openpyxl.
     
