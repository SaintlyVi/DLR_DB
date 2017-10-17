This repository should be cloned as the 'src' directory within a project parent directory. 
To run the scripts without a MSSQL database installation of the DLR project, access to a protected data repository is required. 
You can request access to this repo from @saintlyvi wiebke.toussaint@uct.ac.za

-----

Methodology for DLR DB data processing and anonymising

1. use functions in sqldlr.py to get tables:
	getData() for Answers, profiles, Answers_blob, Answers_char, Answers_number, Questions, Questionaires, Qredundancy, QConstraints, QDataType, LinkTable
	getGroups() for Groups
2. use saveTables() in sqldlr.py to save tables as feather files (for working purposes only, not suitable for archiving)
3. use quSearch(dtype='char') and quSearch(dtype='blob') from answers.py to get all questions
4. save qu_blob and qu_char as .csv files. Add column to anonymise and mark questions 0(keep) and 1(remove)
