use strict;
use warnings;

my @meshsize = (0.575872,0.432842,0.220368,0.109309,0.0762692,0.0436836);
my $logfolder = "./sphere/maxwell_";
my @degs = (["m0_d0",
             "m1_d0",
             "m2_d0",
             "m3_d0",
             "m4_d0"],
            ["m0_d1",
             "m1_d1",
             "m2_d1",
             "m3_d1",
             "m4_d1"],
            ["m0_d2",
             "m1_d2",
             "m2_d2",
             "m3_d2",
             "m4_d2"],
            ["m0_d3",
             "m1_d3",
             "m2_d3",
             "m3_d3",
             "m4_d3"],
            ["m0_d4",
             "m1_d4",
             "m2_d4",
             "m3_d4",
             "m4_d4"]);

for (my $deg = 0; $deg < 5; $deg++) {
  open(my $fh,">","ssphere_$deg") or die "Can't open file: $!";
  print $fh "#h\tE\tdE\tB\tDim\n";
  for (my $mesh = 0; $mesh < @{$degs[$deg]}; $mesh++) {
    my $logfile = $logfolder."m$mesh"."_d$deg"."_err.log";
    my $error = qx\./a.out $logfile 2\;
    $error =~ /Error E: ([\d\.]+) Error dE: ([\d\.]+) Error B: ([\d\.]+)/;
    print $fh "$meshsize[$mesh]\t$1\t$2\t$3\t";
    open (my $logfh,"<",$logfolder."m$mesh"."_d$deg".".log") or die "Can't open file: $!";
    <$logfh>;
    <$logfh>;
    <$logfh> =~ /System dimension: ([\d]+)/;
    print $fh "$1\n";
  }
}

