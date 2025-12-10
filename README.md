# ORC_Cracker
This program using RNN model to idnetify captcha.
# Warning: Don't using to illegal actions!
# Interpretation
-l: Login page that you want crack, example: http://www.example.com/login  
-u: Username.  
-p: Your's path of wordlists.  
-v: Verbse, default is False. If you setting True, every captcha infomation and password will print on the console.  
-t: Threads, default is 10.  
-m: Your's path route of model.  
# Dir construction
ORC_Cracker.py  -main program.  
fake_header.py  -generate fake HTTP headers.  
**wordlist**  
small.txt       -a small password dictionary.  
**captcha**  
This dir is used to save captcha temporarily in attack process.  
**model**  
CNN_RNN_0073.keras  -a model be trained by TensorFlow.  
**config**  
User_Agent.conf     -random select UA header.  
**TestServer**  
A flask server, you can exploit it to test cracker program.  
captcha_creater.py    -using a changed captcha package to generate random captcha.  
**ModelTrain**  
The step of training model.
