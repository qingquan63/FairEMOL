function auc = roc_curve(deci,label_y, plot_flag) %%deci=wx+b, label_y, true label
%     [val,ind] = sort(deci,'descend');
%     roc_y = label_y(ind);
%     stack_x = cumsum(roc_y == 0)/sum(roc_y == 0);
%     stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
%     auc = sum((stack_x(2:length(roc_y))-stack_x(1:length(roc_y)-1)).*stack_y(2:length(roc_y)));
 
    %Comment the above lines if using perfcurve of statistics toolbox
    [stack_x,stack_y,thre,auc]=perfcurve(label_y,deci,1);
    if plot_flag
        plot(stack_x,stack_y);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(['ROC curve of (AUC = ' num2str(auc) ' )']);
    end
end