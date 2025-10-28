package app;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
public class SpringApp {
  public static void main(String[] args) {
    SpringApplication.run(SpringApp.class, args);
  }

  @RestController
  static class HelloController {
    @GetMapping("/")
    String hello() { return "hello, SpringApp"; }
  }
}

