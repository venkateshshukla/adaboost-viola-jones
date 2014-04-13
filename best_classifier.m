function [ best_x, min_error, polarity ] = best_classifier( mydata,weights )



axis([-20 20 -20 20]);



% it will return the minimum error. and the threshold which provides the
% best classification.
yy=-20:0.1:20;


%mydata(1,:) = data(1,:);
%mydata(2:3,:) = data(3:4,:);
%choose a weight vector to start with.
w=[1,1];
no_iter=0;
while no_iter<50

    
    no_iter=no_iter+1;
    x=-w(2)/w(1);
    result=sign(w*mydata(1:2,:));
    
    %plot(x,yy,'LineWidth',10)
    line([x x],[yy(1) yy(end)],'LineWidth',0.5);
    output_pos_index=find(result==1);
    output_neg_index=find(result==-1);

    %get T-, T+, S-, S+
    
    %T_minus = sum(weights(mydata(3,:)==-1));
    %T_plus = sum(weights(mydata(3,:)==+1));
    %less=find(mydata(1,:)<x);
    %S_minus = sum(weights(mydata(3,less)==-1 ));
    %S_plus = sum(weights(mydata(3,less)==+1 ));
    
    
    
    
    a=find(mydata(3,:)==-1);
    b=find(mydata(3,:)==+1);
    T_minus = sum(weights(a));
    T_plus =  sum(weights(b));
    lessm=find(mydata(1,a)<x);
    S_minus=sum(weights(a(lessm)));
    lessp=find(mydata(1,b)<x);
    S_plus=sum(weights(a(lessp)));
    
    
    %find the error.
    [error,polarity]=min([S_plus+(T_minus-S_minus), S_minus+(T_plus-S_plus)]);
    error_list(no_iter,1)=error;
    weight_list(no_iter,1:2)=w(1:2);
    
    
    
    
    %pick a random misclassified point which has the maximum weight.
    
    k=find(result~=mydata(3,:));
    [~,index_max_missclassified]=(max(weights(k)));
    mscfd=k(index_max_missclassified);
    mscfd_data=mydata(1:2,mscfd);
    y=mydata(3,mscfd);
    w=w+(y*mscfd_data)';

    x=-w(2)/w(1);
    %pause(0.25);
    
    
end

[min_error,min_error_index]=min(error_list);
w=weight_list(min_error_index,:);
best_x=-w(2)/w(1);


line([best_x best_x],[yy(1) yy(end)],'LineWidth',4);



    
    
    
end