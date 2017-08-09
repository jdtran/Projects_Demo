drop database if exists GenPreq;
create database GenPreq;
use GenPreq;

CREATE TABLE Course (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL
);

create table Student (
   id int primary key auto_increment,
   firstName varchar(50) not null,
   lastName varchar(50) not null
);

create table Prereq (
   dependent int,
   requirement int,
   foreign key (dependent) references Course(id),
   foreign key (requirement) references Course(id),
   primary key(dependent, requirement)
);

create table Enrollment (
   studentId int,
   courseId int,
   foreign key (studentId) references Student(id),
   foreign key (courseId) references Course(id),
   primary key(studentId, courseId)
);

create table MissingPrereq (
   studentId int,         # Student missing the prereq
   courseId int,          # Course for which the prereq is needed
   prereqId int,          # Course that is needed to satisfy the prereq
   constraint FKMissingPrereq_studentId foreign key (studentId) references Student(id),
   constraint FKMissingPrereq_courseId foreign key (courseId) references Course(id),
   constraint FKMissingPrereq_prereqId foreign key (prereqId) references Course(id)
);

INSERT INTO Course VALUES (1,'Course000'),(2,'Course001'),(3,'Course002'),(4,'Course003'),(5,'Course004'),(6,'Course005'),(7,'Course006'),(8,'Course007'),(9,'Course008'),(10,'Course009'),(11,'Course010'),(12,'Course011'),(13,'Course012'),(14,'Course013'),(15,'Course014'),(16,'Course015'),(17,'Course016');

INSERT INTO Student VALUES (1,'First000','Last000'),(2,'First001','Last001'),(3,'First002','Last002'),(4,'First003','Last003'),(5,'First004','Last004'),(6,'First005','Last005'),(7,'First006','Last006'),(8,'First007','Last007'),(9,'First008','Last008'),(10,'First009','Last009'),(11,'First010','Last010'),(12,'First011','Last011'),(13,'First012','Last012'),(14,'First013','Last013'),(15,'First014','Last014'),(16,'First015','Last015'),(17,'First016','Last016'),(18,'First017','Last017'),(19,'First018','Last018'),(20,'First019','Last019'),(21,'First020','Last020'),(22,'First021','Last021'),(23,'First022','Last022'),(24,'First023','Last023'),(25,'First024','Last024'),(26,'First025','Last025'),(27,'First026','Last026'),(28,'First027','Last027'),(29,'First028','Last028');

INSERT INTO Enrollment VALUES (1,1),(2,1),(4,1),(5,1),(7,1),(8,1),(10,1),(11,1),(13,1),(14,1),(16,1),(17,1),(18,1),(19,1),(21,1),(22,1),(24,1),(25,1),(27,1),(28,1),(2,2),(4,2),(5,2),(7,2),(8,2),(10,2),(11,2),(13,2),(14,2),(16,2),(17,2),(19,2),(21,2),(22,2),(24,2),(25,2),(27,2),(28,2),(3,3),(6,3),(9,3),(12,3),(15,3),(20,3),(23,3),(26,3),(29,3),(4,4),(5,4),(7,4),(8,4),(10,4),(11,4),(13,4),(14,4),(16,4),(17,4),(21,4),(22,4),(24,4),(25,4),(27,4),(28,4),(5,5),(7,5),(8,5),(10,5),(11,5),(13,5),(14,5),(16,5),(17,5),(22,5),(24,5),(25,5),(27,5),(28,5),(6,6),(9,6),(12,6),(15,6),(23,6),(26,6),(29,6),(7,7),(8,7),(10,7),(11,7),(13,7),(14,7),(16,7),(17,7),(24,7),(25,7),(27,7),(28,7),(8,8),(10,8),(11,8),(13,8),(14,8),(16,8),(17,8),(25,8),(27,8),(28,8),(9,9),(12,9),(15,9),(26,9),(29,9),(10,10),(11,10),(13,10),(14,10),(16,10),(17,10),(27,10),(28,10),(11,11),(13,11),(14,11),(16,11),(17,11),(28,11),(12,12),(15,12),(29,12),(13,13),(14,13),(16,13),(17,13),(14,14),(16,14),(17,14),(15,15),(16,16),(17,16),(17,17);

INSERT INTO Prereq VALUES (2,1),(5,1),(8,1),(11,1),(14,1),(17,1),(4,2),(7,2),(10,2),(13,2),(16,2),(6,3),(9,3),(12,3),(15,3),(5,4),(8,4),(11,4),(14,4),(17,4),(7,5),(10,5),(13,5),(16,5),(9,6),(12,6),(15,6),(8,7),(11,7),(14,7),(17,7),(10,8),(13,8),(16,8),(12,9),(15,9),(11,10),(14,10),(17,10),(13,11),(16,11),(15,12),(14,13),(17,13),(16,14),(17,16);

SELECT * FROM Student;
SELECT * FROM Course;
SELECT * FROM Prereq ORDER BY dependent;
SELECT * FROM Enrollment;

SELECT GROUP_CONCAT(courseId ORDER BY courseID SEPARATOR ' ') FROM Enrollment WHERE studentId = 17 GROUP BY studentId;
SELECT dependent, GROUP_CONCAT(requirement ORDER BY requirement SEPARATOR ' ') FROM Prereq GROUP BY dependent;

SELECT * FROM MissingPrereq;