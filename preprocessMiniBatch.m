function [X,features,Y] = preprocessMiniBatch(XCell,featureCell,YCell)
    
    % Extract transit predictors from cell and concatenate.
    X = cat(4,XCell{:});
    % Extract 6 features data from cell and concatenate.
    features = cat(1,featureCell{:})';
    % Extract target data from cell and concatenate.
    Y = cat(1,YCell{:})';        
end