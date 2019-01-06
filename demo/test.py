import argparse
parser = argparse.ArgumentParser()

group1 = parser.add_argument_group('group1')
group1.add_argument('--test1', help="test1")

group2 = parser.add_argument_group('group2')
group2.add_argument('--test2', help="test2")

args = parser.parse_args('--test1 one --test2 two'.split())

# print(args)

# args = parser.parse_args()

print(parser._action_groups)
arg_groups={}

for group in parser._action_groups:
    group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    print(group_dict)
    arg_groups[group.title]=argparse.Namespace(**group_dict)

print(arg_groups['group1'].test1)