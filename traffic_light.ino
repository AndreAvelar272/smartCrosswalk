#include <Adafruit_NeoPixel.h>
#include <WiFi.h>

// Set web server port number to 80
WiFiServer server(80);
// Variable to store the HTTP request
String header;

// Current time
unsigned long currentTime = millis();
// Previous time
unsigned long previousTime = 0; 
// Define timeout time in milliseconds (example: 2000ms = 2s)
const long timeoutTime = 2000;

String lightState = "green";

#define PIN        2
#define NUMPIXELS 6

Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);
#define DELAYVAL 500

// Replace with your network credentials
const char* ssid = "<change this to the AP/hotspot name>";
const char* password = "<change this to its password>";

void initWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }
  Serial.println(WiFi.localIP());
}


void setup() {

  pixels.begin();
  pixels.setBrightness(255);
  Serial.begin(115200);

  displayInit();
  Serial.println("Starting wifi connection process");
  initWiFi();

  server.begin();
  Serial.print("RRSI: ");
  Serial.println(WiFi.RSSI());
  displayConnected();
  displayConnected();
  displayConnected();

}

void loop(){
  WiFiClient client = server.available();   // Listen for incoming clients

  if (client) {                             // If a new client connects,
    currentTime = millis();
    previousTime = currentTime;
    Serial.println("New Client.");          // print a message out in the serial port
    String currentLine = "";                // make a String to hold incoming data from the client
    while (client.connected() && currentTime - previousTime <= timeoutTime) {  // loop while the client's connected
      currentTime = millis();
      if (client.available()) {             // if there's bytes to read from the client,
        char c = client.read();             // read a byte, then
        Serial.write(c);                    // print it out the serial monitor
        header += c;
        if (c == '\n') {                    // if the byte is a newline character
          // if the current line is blank, you got two newline characters in a row.
          // that's the end of the client HTTP request, so send a response:
          if (currentLine.length() == 0) {
            // HTTP headers always start with a response code (e.g. HTTP/1.1 200 OK)
            // and a content-type so the client knows what's coming, then a blank line:
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println("Connection: close");
            client.println();
            
            // turns the GPIOs on and off
            if (header.indexOf("GET /green/on") >= 0) {
              Serial.println("turning the light green");
              changeGreen();
              lightState = "green";
            } else if (header.indexOf("GET /red/on") >= 0) {
              Serial.println("turning the light red");
              changeRed();
              lightState = "red";
            } else if (header.indexOf("GET /yellow/on") >= 0) {
              Serial.println("turning the light yellow");
              changeYellow();
              lightState = "yellow";
            }
            
            // Display the HTML web page
            client.println("<!DOCTYPE html><html>");
            client.println("<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
            client.println("<link rel=\"icon\" href=\"data:,\">");
            // CSS to style the on/off buttons 
            // Feel free to change the background-color and font-size attributes to fit your preferences
            client.println("<style>html { font-family: Helvetica; display: inline-block; margin: 0px auto; text-align: center;}");
            client.println(".button { background-color: #4CAF50; border: none; color: white; padding: 16px 40px;");
            client.println("text-decoration: none; font-size: 30px; margin: 2px; cursor: pointer;}");
            client.println(".button2 {background-color: #555555;}</style></head>");
            
            // Web Page Heading
            client.println("<body><h1>ESP32 Web Server</h1>");
            
            // Display current state, and ON/OFF buttons for GPIO 26  
            client.println("<p>Light State " + lightState + "</p>");
            // If the output26State is off, it displays the ON button       
            client.println("<p><a href=\"/green/on\"><button class=\"button\">turnGreen</button></a></p>");
            client.println("<p><a href=\"/red/on\"><button class=\"button\">turnRed</button></a></p>");
            client.println("<p><a href=\"/yellow/on\"><button class=\"button\">turnYellow</button></a></p>");
            client.println("</body></html>");
            
            // The HTTP response ends with another blank line
            client.println();
            // Break out of the while loop
            break;
          } else { // if you got a newline, then clear currentLine
            currentLine = "";
          }
        } else if (c != '\r') {  // if you got anything else but a carriage return character,
          currentLine += c;      // add it to the end of the currentLine
        }
      }
    }
    // Clear the header variable
    header = "";
    // Close the connection
    client.stop();
    Serial.println("Client disconnected.");
    Serial.println("");
  }
}

void displayInit(){

  changeGreen();
  delay(1000);
  changeRed();
  delay(1000);
  changeYellow();
  delay(1000);

}

void displayConnected(){
  pixels.clear();
  changeGreen();
  delay(500);
  pixels.clear();
  changeGreen();

  pixels.clear();
  changeRed();
  delay(500);
  pixels.clear();
  changeRed();

  pixels.clear();
  changeYellow();
  delay(500);
  pixels.clear();
  changeYellow();
  delay(500);
  pixels.clear();
}

void changeGreen(){
  pixels.clear();  
  pixels.setPixelColor(4, pixels.Color(0, 255, 0));
  pixels.setPixelColor(5, pixels.Color(0, 255, 0));
  pixels.show();
}


void changeYellow(){
  pixels.clear();  
  pixels.setPixelColor(2, pixels.Color(255, 255, 0));
  pixels.setPixelColor(3, pixels.Color(255, 255, 0));
  pixels.show();
}
void changeRed(){
  pixels.clear();  
  pixels.setPixelColor(0, pixels.Color(255, 0, 0));
  pixels.setPixelColor(1, pixels.Color(255, 0, 0));
  pixels.show();
}
