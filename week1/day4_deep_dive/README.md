/predict vs /predict-slow

These are the two endpoiunt we used in lifespam class 
/predict - it uses the loaded model it does not have to load model each time so it take less time to response.
/predict-slow - it load model each time so it take more time than /predict endpoint. The best practice it to load model is at starting of lifespam so each endpoint does not have to load model each call.

what is the difference between a path parameter and a query parameter? Give one example URL for each.

path perameter is always required in the URL. it should be uniqe. it is part of URL.
query parameter is optional for the URL and after ? it append.


yield  -  yield is act as transition point between application starting point and shutdown. code writen before yield will run once before server started, code writen after yield run when server stop. 