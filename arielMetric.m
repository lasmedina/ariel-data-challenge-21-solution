function score = arielMetric(prediction,target)
% See: https://www.ariel-datachallenge.space/ML/documentation/scoring

if ~iscell(prediction)
    prediction = prediction';
    p = cell(length(target),1);
    for i = 1:length(target)
        p{i} = prediction(i,:);
    end
    prediction = p';
end

C = cellfun(@minus,prediction',target,'Un',0);
C = cellfun(@abs,C,'UniformOutput',0);
C = cellfun(@(x,y) 2*x.*y, target, C, 'UniformOutput',false);
C = cellfun(@sum, C, 'UniformOutput',false);

numerator = sum(cell2mat(C));
num_of_elements = cellfun(@numel, target);
denominator = sum(num_of_elements);

score = 1e4 - (numerator * 1e6 / denominator);
end

