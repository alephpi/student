parser([]) --> [].
parser(Tree) --> assign(Tree).

assign([assignment, ident(X), '=', Exp]) --> id(X), [=], expr(Exp), [;].

id(X) --> [X], { atom(X) }.

expr([expression, Term]) --> term(Term).
expr([expression, Term, Op, Exp]) --> term(Term), add_sub(Op), expr(Exp).

term([term, F]) --> factor(F).
term([term, F, Op, Term]) --> factor(F), mul_div(Op), term(Term).

factor([factor, int(N)]) --> num(N).
factor([factor, Exp]) --> ['('], expr(Exp), [')'].

add_sub(Op) --> [Op], { memberchk(Op, ['+', '-']) }.
mul_div(Op) --> [Op], { memberchk(Op, ['*', '/']) }.

num(N) --> [N], { number(N) }.