clear all
close all
keep_spars_idx = [5 8 11];
errorsm10 = table2array(readtable('error_snr_-10.csv'));
spars = errorsm10(keep_spars_idx,1);
errorsm10 = errorsm10(keep_spars_idx,2:end);
errors0 = table2array(readtable('error_snr_0.csv'));
errors0 = errors0(keep_spars_idx,2:end);
errors10 = table2array(readtable('error_snr_10.csv'));
errors10 = errors10(keep_spars_idx,2:end);
errors20 = table2array(readtable('error_snr_20.csv'));
errors20 = errors20(keep_spars_idx,2:end);
errors30 = table2array(readtable('error_snr_30.csv'));
errors30 = errors30(keep_spars_idx,2:end);
fourier_compl = [errorsm10(:,1),errors0(:,1),errors10(:,1),errors20(:,1),errors30(:,1)]';
sep_compl = [errorsm10(:,2),errors0(:,2),errors10(:,2),errors20(:,2),errors30(:,2)]';
slep4_compl = [errorsm10(:,3),errors0(:,3),errors10(:,3),errors20(:,3),errors30(:,3)]';
snr = [-10, 0, 10, 20, 30];
plot(snr,10*log10(sep_compl),'LineWidth',6);hold on; plot(snr,10*log10(slep4_compl),'LineWidth',6)
legend('Sep WL 60','Sep WL 90','Sep WL 210','Slepians 60','Slepians 90','Slepians 210')