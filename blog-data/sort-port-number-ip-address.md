---
title: "sort port number ip address?"
date: "2024-12-13"
id: "sort-port-number-ip-address"
---

 I get it You're asking how to sort a list of network addresses based on port number and then IP address That's a pretty common task especially when dealing with network monitoring or log analysis I've wrestled with this beast more times than I care to admit let me tell you about my own suffering

 so first let's talk about the data format I’m assuming we have some kind of list or array each entry containing both the IP and the port typically formatted as strings like "1921681100080" or "10101500022" Notice I’m using the simplest representation possible for the sake of clarity in code examples we might have a more complex data structure but for sorting purposes string representation is fine

Back in my early days building network monitoring tools one of my first projects needed this exact functionality I had thousands of these address strings coming in from different sources I needed to sort them to identify services that use specific ports and correlate them to IP addresses It wasn’t pretty initial code was super inefficient but I learned from that experience you always learn from errors

My first attempt was using a very basic implementation trying to extract port and then IP for comparison it was a mess of string slicing and number conversions slow as molasses on a cold day don't even ask I'm not showing that code no one should see that horrible thing

Anyway moving on The key to efficient sorting in this case is having a good comparison function Python or JavaScript both offer this capability in their sort methods So I will show both Python and JS examples and since I am an older guy with preference for the language I will add a bonus C++ example too

**Python Example**

```python
def sort_ip_port(address_list):
    def compare_addresses(address1, address2):
      ip1, port1 = address1.rsplit(':', 1)
      ip2, port2 = address2.rsplit(':', 1)
      port1 = int(port1)
      port2 = int(port2)

      if port1 < port2:
          return -1
      elif port1 > port2:
          return 1
      else:
           ip_parts1 = list(map(int, ip1.split('.')))
           ip_parts2 = list(map(int, ip2.split('.')))

           for i in range(4):
               if ip_parts1[i] < ip_parts2[i]:
                    return -1
               elif ip_parts1[i] > ip_parts2[i]:
                  return 1
           return 0

    return sorted(address_list, key=lambda x: (int(x.rsplit(':', 1)[1]), list(map(int,x.rsplit(':',1)[0].split('.')))))
     
addresses = ["192.168.1.10:80", "10.0.0.5:22", "192.168.1.5:8080", "10.0.0.1:22", "192.168.1.10:22", "192.168.1.10:8080", "10.0.0.2:22"]
sorted_addresses = sort_ip_port(addresses)
print(sorted_addresses)
#Output: ['10.0.0.1:22', '10.0.0.2:22', '10.0.0.5:22', '192.168.1.10:22', '192.168.1.10:80', '192.168.1.5:8080', '192.168.1.10:8080']

```
In this code, we use a lambda function for direct comparison within the sorted call This extracts the port converts to integer and constructs the tuple used for sorting It’s cleaner I found when I did more network stuff for an older project but I remember I had to do this again from scratch because for some reason in the last company I worked at for a short time I had to write all from scratch which was a waste of resources But I did learn another way of doing this as I will show next in JS

**JavaScript Example**
```javascript
function sortIPAndPort(addresses) {
    return addresses.sort((a, b) => {
      const [ipA, portA] = a.split(":");
      const [ipB, portB] = b.split(":");

      const portIntA = parseInt(portA, 10);
      const portIntB = parseInt(portB, 10);

      if (portIntA < portIntB) {
        return -1;
      }
      if (portIntA > portIntB) {
        return 1;
      }

      const ipPartsA = ipA.split(".").map(Number);
      const ipPartsB = ipB.split(".").map(Number);

      for (let i = 0; i < 4; i++) {
        if (ipPartsA[i] < ipPartsB[i]) {
            return -1;
        }
        if (ipPartsA[i] > ipPartsB[i]) {
          return 1;
        }
      }
        return 0;
    });
}

const addresses = ["192.168.1.10:80", "10.0.0.5:22", "192.168.1.5:8080", "10.0.0.1:22", "192.168.1.10:22", "192.168.1.10:8080", "10.0.0.2:22"];
const sortedAddresses = sortIPAndPort(addresses);
console.log(sortedAddresses);
//Output: ['10.0.0.1:22', '10.0.0.2:22', '10.0.0.5:22', '192.168.1.10:22', '192.168.1.10:80', '192.168.1.5:8080', '192.168.1.10:8080']
```
This JavaScript example is a direct translation of the Python example and shows that the logic and concept is portable between languages Both examples do the same thing first compares the ports then if the ports are equal compares the IP address

And as a bonus here's a C++ example since sometimes you need that raw performance particularly when you are dealing with systems software

**C++ Example**
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

using namespace std;
vector<int> splitIP(const string& ip){
  vector<int> parts;
  stringstream ss(ip);
  string part;
  while(getline(ss,part,'.')){
    parts.push_back(stoi(part));
  }
  return parts;
}
int compareAddresses(const string& a,const string& b){
   size_t posA= a.rfind(":");
   size_t posB= b.rfind(":");
   int portA=stoi(a.substr(posA+1));
   int portB=stoi(b.substr(posB+1));
   string ipA=a.substr(0,posA);
   string ipB=b.substr(0,posB);
   if (portA<portB){
     return -1;
    } else if (portA > portB){
        return 1;
    } else {
      vector<int> partsA=splitIP(ipA);
      vector<int> partsB=splitIP(ipB);
      for(size_t i=0;i<4;i++){
          if(partsA[i]<partsB[i]){
              return -1;
          }else if(partsA[i]>partsB[i]){
              return 1;
          }
      }
    }
    return 0;
}
int main() {
    vector<string> addresses = {"192.168.1.10:80", "10.0.0.5:22", "192.168.1.5:8080", "10.0.0.1:22", "192.168.1.10:22", "192.168.1.10:8080", "10.0.0.2:22"};
     sort(addresses.begin(),addresses.end(), compareAddresses);

     for(const auto& address : addresses){
        cout<<address<<endl;
     }
    return 0;
}
//Output:
//10.0.0.1:22
//10.0.0.2:22
//10.0.0.5:22
//192.168.1.10:22
//192.168.1.10:80
//192.168.1.5:8080
//192.168.1.10:8080

```
This C++ example, again does the same using a custom comparator function to sort the vector

Now let's talk resources you mentioned no links so let's do that properly if you're serious about this network stuff there are some good books and papers you should check out

For network fundamentals "Computer Networking: A Top-Down Approach" by Kurose and Ross is a must-read It provides a solid understanding of protocols and architectures For more advanced techniques maybe look at "TCP/IP Illustrated" by Stevens it's a classic text that delves into the intricacies of network communication It will make you a professional

And if you want to deep dive into the data structures and algorithms behind sorting operations a good book is "Introduction to Algorithms" by Cormen Leiserson Rivest and Stein It is the bible when dealing with algorithms or searching for proper implementations it is a must in every developer library

You know debugging this kind of code it's like trying to find a specific grain of sand on a beach especially if the input data is messed up Once I spent two full days because the logs had an space character and I was splitting based on colons... never assume anything that's what I learnt the hard way that day lol

Anyways thats what I recommend for the problem I’ve shown you three different examples and I hope this helps you solve your address sorting problem let me know if you have more questions
