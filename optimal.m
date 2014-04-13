function [ best_x, min_error, polarity ] = optimal( mydata,weights )

    [data,index]=sort(mydata(1,:));
    y=mydata(3,index);
    weights1=weights(index);
    
    
    total_pos_idx = find(y==+1);
    total_neg_idx = find(y==-1);
    
    T_plus = sum(weights1(total_pos_idx));
    T_minus = sum(weights1(total_neg_idx));
    
     for i = 1:length(mydata)
        
       S_plus = sum(weights1(find(i>total_pos_idx)));
       S_minus = sum(weights1(find(i>total_neg_idx)));
       
       [error_list(i),polarity(i)]=min([S_plus+(T_minus - S_minus), S_minus+(T_plus - S_plus)]);
        
     end
     
     
     [min_error,min_error_idx] = min(error_list);
     best_x=data(min_error_idx);
     polarity=polarity(min_error_idx);
     
     
     axis([-20 20 -20 20]);
     yy=-20:0.1:20;
     line([best_x best_x],[yy(1) yy(end)],'LineWidth',4);





end