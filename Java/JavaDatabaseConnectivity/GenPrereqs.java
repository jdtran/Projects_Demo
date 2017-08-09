/**
 * -----------------------------------------------------------------------
 * @author  John Tran
 * @version 20 May, 2016
 * @class   CPE-365  Databases
 * @description Automating SQL queries and insertion with MySQL connector.
 *              This program lists all the immediate pre-reqs required 
 *              for a requested class and enrolls the student if
 *              the student meets all the requirements.          
 * -----------------------------------------------------------------------
 */
 
import java.sql.*;
import java.util.*;
import java.util.Map.Entry;

public class GenPrereqs {
   static final int TABLE_MAX = 4;
   private static List<Student> std = new ArrayList<Student>();
   private static List<Course> crs = new ArrayList<Course>();
   private static Map<Integer, HashSet<Integer>> reqMap = 
                  new HashMap<Integer, HashSet<Integer>>();

   static public class Course {
      int id;
      String name;

      public Course(int i, String n) {
          id = i;
          name = n;
      }
   }
   
   static public class Student {
      int id;
      String first, last;
      Set<Integer> coursesTaken;
      
      public Student(int i, String f, String l) {
          id = i;
          first = f;
          last = l;
          coursesTaken = new HashSet<Integer>();
      }
   }
   
   static void closeEm(Object... toClose) {
      for (Object obj: toClose)
         if (obj != null)
            try {
               obj.getClass().getMethod("close").invoke(obj);
            } catch (Throwable t) {System.out.println("Log bad close");}
   }
   
   static void FillAll() {
      Iterator<Integer> iter;
      int i = 0, size = 0;
      
      for (Map.Entry<Integer, HashSet<Integer>> entry : reqMap.entrySet()) {
         HashSet<Integer> prereqs = entry.getValue();    
         iter = prereqs.iterator();
         while (iter.hasNext()) {
            size = prereqs.size();
            i = iter.next();
            if (reqMap.get(i) != null) {
               prereqs.addAll(reqMap.get(i));
               if (size != prereqs.size())
               iter = prereqs.iterator();
            }
         }
      }
   }
   
   static void PairStudentCourse() {
      int j = 0;
      for (int i = 0; i < std.size(); i++) {
         std.get(i).coursesTaken.add(crs.get(j).id);
         std.get(i).coursesTaken.addAll(reqMap.get(crs.get(j).id));
         j++;
         
         if (j >= crs.size()) {
            j = 0;
         }
      }
   }
   
   static void GenEnroll(Connection cnc) {
      Statement stm = null;
      PairStudentCourse();
      
      try {
         for(Student s : std) {
            for(int i : s.coursesTaken) { 
               stm = cnc.createStatement();
               String insertTableSQL = "INSERT INTO Enrollment"
                     + "(studentId, courseId) " + "VALUES"
                     + "('" + s.id + "'" + ", '" + i + "')";
               stm = cnc.createStatement();
               stm.executeUpdate(insertTableSQL);
            }
         }
      }
      catch (SQLException e) {
         System.out.println(e.getMessage());
      }
      finally {
         closeEm(stm);
      }
   }
   
   static void GenPrereq(Connection cnc, int freq) {
      Statement stm = null;
      try {
         int idi, idj;
         stm = cnc.createStatement();
         for (int i = 0; i < crs.size(); i++) {
            idi = crs.get(i).id;
            HashSet<Integer> prereqId = new HashSet<Integer>();;

            for (int j = 0; j < crs.size(); j++) {
               idj = crs.get(j).id;

               if (((idi + idj) * 8191 % freq == 0) && idi > idj){
                  String insertTableSQL = "INSERT INTO Prereq"
                        + "(dependent, requirement) " + "VALUES"
                        + "('" + idi + "'" + ", '" + idj + "')";                 
                  stm = cnc.createStatement();
                  stm.executeUpdate(insertTableSQL);
                  prereqId.add(idj);
               }
            }
            reqMap.put(idi, prereqId);
         }
      }
      catch (SQLException e) {
         System.out.println(e.getMessage());
      }
      finally {
         closeEm(stm);
      }
   }
   
   static void GenStudents(Connection cnc, int numStudents) {
      Statement stm = null;
      String first, last = "";
      int ndx = 0, genKey = 0;  

      try {
         while(ndx < numStudents) {
            first = String.format("First%03d", ndx);
            last = String.format("Last%03d", ndx);

            String insertTableSQL = "INSERT INTO Student"
                  + "(firstName, lastName) " + "VALUES"
                  + "('" + first + "', '" + last + "')";         
            stm = cnc.createStatement();
            stm.executeUpdate(insertTableSQL, 
                              Statement.RETURN_GENERATED_KEYS);
            ResultSet rs = stm.getGeneratedKeys();
            
            if (rs.next()){
               genKey=rs.getInt(1);
            }
            
            std.add(new Student(genKey, first, last));
            ndx++;
         }
      }
      catch (SQLException e) {
         System.out.println(e.getMessage());
      }
      finally {
         closeEm(stm);
      }
   }
   
   public static void GenCourses(Connection cnc, int numCourse) {
      Statement stm = null;
      String course = "";
     
      int ndx = 0, genKey = 0;    

      try {
         while(ndx < numCourse) {
            course = String.format("Course%03d", ndx);

            String insertTableSQL = "INSERT INTO Course"
                  + "(name) " + "VALUES"
                  + "('" + course + "')";
            stm = cnc.createStatement();
            stm.executeUpdate(insertTableSQL, 
                              Statement.RETURN_GENERATED_KEYS);
            ResultSet rs = stm.getGeneratedKeys();
            if (rs.next()){
               genKey=rs.getInt(1);
            }
            
            crs.add(new Course(genKey, course));
            ndx++;
         }
      }
      catch (SQLException e) {
         System.out.println(e.getMessage());
      }
      finally {
         closeEm(stm);
      }
   }
   
   public static void EmptyTables(Connection cnc) {
      int ndx = 0;
      Statement stmt = null;
      String[] del = new String[]{"DELETE FROM Enrollment",
                                  "DELETE FROM Prereq",
                                  "DELETE FROM Course", 
                                  "DELETE FROM Student"};
      try {
         while (ndx < TABLE_MAX) {
            stmt = cnc.createStatement();           
            stmt.executeUpdate(del[ndx]);
            ndx++;
         }
      }          
      catch(SQLException s){
         System.out.println(s.getMessage());
      }
   }

   public static void Exec(Connection cnc, Scanner in) 
                      throws NumberFormatException, SQLException {
      String numCourses, numStudents, freq;      
      String elem = in.nextLine();
      String[] items = elem.split(" ");
      
      if (items.length == TABLE_MAX) {
         EmptyTables(cnc);
      }

      numCourses = items[0];
      numStudents = items[1];
      freq = items[2];
      
      GenStudents(cnc, Integer.parseInt(numStudents));
      GenCourses(cnc, Integer.parseInt(numCourses));
      GenPrereq(cnc, Integer.parseInt(freq));
      FillAll();
      GenEnroll(cnc);
   }
   
   public static void Login(String dbRes, String login, String pwd) {
      Connection cnc = null;
      Scanner in = null;
      try {
         cnc = DriverManager.getConnection(dbRes, login, pwd);
         in = new Scanner(System.in); 
         Exec(cnc, in);
      }
      catch (SQLException err) {
         System.out.println(err.getMessage());
      }
      finally {
         closeEm(cnc, in);
      }
   }
   
   public static void main(String[] args) {
      Login(args[0], args[1], args[2]);
   }
}
