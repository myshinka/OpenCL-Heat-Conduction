clc
clear

%% read file

heat_con_raw = readtable('heat_con.csv');

begin = 1;
tsteps = 100;  %only this one needs to be changed manually
ni = width(heat_con_raw) - 1;
nj = height(heat_con_raw)/tsteps;

for n = 1:tsteps
    heat_array = table2array( heat_con_raw(begin:(begin+ni-1),1:nj) );
    heat_array(end, end, 2) = 0;
    heat_array(end, end, 3) = 0;
    heat_con_sep{n} = heat_array./100;
    begin = begin + ni;
end

%% make frames

filename = 'heat_con.gif';
for n = 1:tsteps
      [imind,cm] = rgb2ind(heat_con_sep{n},65536);
      if n == 1
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf,"DelayTime",0.1);
      else
          imwrite(imind,cm,filename,'gif','WriteMode','append',"DelayTime",0.1);
      end
end