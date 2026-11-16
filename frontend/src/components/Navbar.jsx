import { Link, useLocation } from 'react-router-dom';
import { Brain } from 'lucide-react';

function Navbar() {
  const location = useLocation();
  
  const isActive = (path) => location.pathname === path ? 'active' : '';
  
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <Brain size={28} />
        <h1>OX-MABSR</h1>
        <span>AAAI 2026</span>
      </Link>
      <div className="navbar-links">
        <Link to="/" className={isActive('/')}>Home</Link>
        <Link to="/demo" className={isActive('/demo')}>Demo</Link>
        <Link to="/about" className={isActive('/about')}>About</Link>
        <Link to="/demo" className="btn-primary">Try It Now</Link>
      </div>
    </nav>
  );
}

export default Navbar;
