#!/usr/bin/perl
require 5.006001;
use strict;
use warnings;
use ExtUtils::MakeMaker;
WriteMakefile(
    NAME         => 'Statistics::RankCorrelation',
    VERSION_FROM => 'lib/Statistics/RankCorrelation.pm',
    PREREQ_PM    => {},
    ($] >= 5.005
        ? (ABSTRACT_FROM => 'lib/Statistics/RankCorrelation.pm',
           AUTHOR        => 'Gene Boggs <gene@cpan.org>')
        : ()
    ),
);
