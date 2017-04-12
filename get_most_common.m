most_common = importdata('google_10000_english.txt');
N = 100;
most_common = most_common(1:N);
save([num2str(N),'_most_common_words'],'most_common');