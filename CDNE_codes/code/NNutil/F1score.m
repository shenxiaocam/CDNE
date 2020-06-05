function [micro,macro]=F1score(trueLabelMatrix,predLabelMatrix)

F = 0;
tp_sum = 0;
fp_sum = 0;
fn_sum = 0;
tn_sum = 0;

num_labels=size(trueLabelMatrix,2);

for i = 1:num_labels  
  orig=trueLabelMatrix(:,i);
  pre=predLabelMatrix(:,i);
  
  tp = full(sum(orig == +1 & pre == +1));
  fn = full(sum(orig == +1 & pre == 0));
  tn = full(sum(orig == 0 & pre == 0));
  fp = full(sum(orig == 0 & pre == +1));
  
  this_F = 0;    
  if tp ~= 0 || fp ~= 0 || fn ~= 0
    this_F = (2*tp) / (2*tp + fp + fn);
  end
  F = F + this_F;

  tp_sum = tp_sum + tp;
  fp_sum = fp_sum + fp;
  fn_sum = fn_sum + fn;
  tn_sum = tn_sum + tn;

%   fprintf(1, 'INFO: label %d:\tF %.6f\ttp %d\tfp %d\ttn %d\tfn %d\n', i, this_F, tp, fp, tn, fn);
end

micro = (2*tp_sum) / (2*tp_sum + fp_sum + fn_sum);
macro = F / num_labels;

% fprintf(1, 'INFO: tp_sum %d fp_sum %d fn_sum %d tn_sum %d\n', tp_sum, fp_sum, fn_sum, tn_sum);
% fprintf(1, 'INFO: microaverage: %.6f\n', micro);
% fprintf(1, 'INFO: macroaverage: %.6f\n', macro);

