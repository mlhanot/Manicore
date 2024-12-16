use strict;
use warnings;

# Parse the computed error for postprocessing
# Compute the rate as a linear regression
my $convfolder = "./ssphere_";

for (my $deg = 0; $deg < 5; $deg++) {
  my @h;
  my @E;
  my @dE;
  my @B;
  open(my $fh,"<","$convfolder$deg") or die "Can't open file: $!";
  <$fh>;
  while (my $line = <$fh>) {
    $line =~ /^([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)/;
    push(@h,$1);
    push(@E,$2);
    push(@dE,$3);
    push(@B,$4);
  }
  print "Degree: $deg\n";
  print "Rate E: " . linearfit(\@h,\@E) . "\t dE: " . linearfit(\@h,\@dE) . "\t B: " . linearfit(\@h,\@B) . "\n";
}

sub linearfit {
  my ($href, $Eref) = @_;
  my $n = @$href;
  my $xb = 0; 
  my $yb = 0;
  for (my $i = 0; $i < $n; ++$i) {
    $xb += log($href->[$i])/$n;
    $yb += log($Eref->[$i])/$n;
  }
  my $num = 0;
  my $den = 0;
  for (my $i = 0; $i < $n; ++$i) {
    $num += (log($href->[$i]) - $xb)*(log($Eref->[$i]) - $yb);
    $den += (log($href->[$i]) - $xb)*(log($href->[$i]) - $xb);
  }
  return $num/$den;
}
