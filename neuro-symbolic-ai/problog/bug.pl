.2::raining.
.999::alice_is_happy.
.05::school_test.
.1::fun_class.
.3::alice_is_happy :- fun_class.
% .7::\+alice_is_happy :- raining.
% .9::\+alice_is_happy :- school_test.
% Knowing that the class was fun...
% evidence(fun_class).
% ...what is the probability that Alice is happy?
query(alice_is_happy).