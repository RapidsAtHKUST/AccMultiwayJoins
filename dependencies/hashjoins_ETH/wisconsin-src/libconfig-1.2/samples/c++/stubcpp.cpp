
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "libconfig.h++"

using namespace libconfig;

int main(int argc, char **argv)
  {
  Config cfg;

  try
    {
    FILE *fp = fopen("main.cfg", "r");

    if(! fp)
    {
      printf("Unable to open main.cfg\n");
      exit(1);
    }
    
    cfg.read(fp);
    fclose(fp);
    
//  const ConfigSetting& setting = cfg.lookup("application.window.size.w");
//  long val = setting;
    
    long val = cfg.lookup("application.window.size.w");
    printf("val: %ld\n", val);

    std::string title = cfg.lookup("application.window.title");
    std::cout << "title: " << title << std::endl;

    Setting &ss = cfg.lookup("application.window.title");
    std::string title2 = ss;
    std::cout << "title: " << title2 << std::endl;

    std::string rr = "foo";

    rr = (const char *)cfg.lookup("application.window.title");
//    rr = (std::string)(cfg.lookup("application.window.title"));
    std::cout << "rr: " << rr << std::endl;

    const char *rrr = cfg.lookup("application.window.title");
    
    Setting &s = cfg.lookup("application.group1.my_array");  
    long val4;
    val4 = s[4];
    printf("item #4 is: %ld\n", val4);
    printf("location of my_array is %d\n", s.getSourceLine());

    Setting &grp = cfg.lookup("application.group1.group2");

    Setting &zzz = cfg.lookup("application.group1.group2.zzz");
    printf("location of zzz is at %d\n", zzz.getSourceLine());

    Setting &root = cfg.getRoot();

    Setting &rootn = root.add("new-one-at-top", Setting::TypeGroup);

    Setting &ngp = rootn.add("element",  Setting::TypeFloat);

    Setting &misc = root["misc"];
    unsigned int portnum = 0;
    misc.lookupValue("port", portnum);
    printf("port # is: %d\n", portnum);
    
    ngp = 1.1234567890123;
    
//    long val22 = s[22];
//    printf("item #22 is: %d\n", val22);

    Setting &snew = grp.add("foobar",  Setting::TypeArray);

    snew.add(Setting::TypeInt);
    snew.add(Setting::TypeInt);

    snew.add(Setting::TypeInt);
    snew.add(Setting::TypeInt);
    
    puts("created new array");

    snew[0] = 55;
    puts("elem 0");
    snew[1] = 66;
    puts("elem 1");

    cfg.setAutoConvert(true);

    double dd = cfg.lookup("application.group1.x");
    printf("auto-converted int->double: %f\n", dd);

    int ii = cfg.lookup("misc.pi");
    printf("auto-converted double->int: %d\n", ii);
   
    Setting &sdel = cfg.lookup("application");

    sdel.remove("group1");

    
    Setting &books = cfg.lookup("books");
    puts("found books");
    Setting &book = books.add(Setting::TypeGroup);
    puts("added book");

    Setting &sss = book.add("Title", Setting::TypeString);
    puts("added title");
    
    sss = "Alice in Wonderland";

    Setting &sss2 = book.add("Price", Setting::TypeFloat);
    sss2 = 9.99;

    cfg.write(stdout);

    
    }
  catch(ParseException& ex)
    {
    printf("error on line %d: %s\n", ex.getLine(),
           ex.getError());
    }
  catch(ConfigException& cex)
  {
    printf("config exception!\n");
  }
    
  return(0);
  }
